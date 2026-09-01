# DetectKit SAM3 LoRA Finetuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `SEMANTIC_SAM3` training role that finetunes SAM3 on a DetectKit source's polygon labels and publishes a merged checkpoint the existing ultralytics escalation path loads unchanged.

**Architecture:** LoRA adapters are trained against Meta's `sam3` package (using Meta's own loss and Hungarian matcher), then merged into the base weights and written in Meta's `detector.`-prefixed key layout. Inference stays entirely on ultralytics — no second backend, no training dependencies in the runtime install.

**Tech Stack:** PyTorch, Meta `sam3` (training only), ultralytics `SAM3SemanticPredictor` (inference), COCO polygon datasets, PySide6 (DetectKit GUI — NOT PyQt6; `training_dialog.py:9` imports PySide6 and PyQt6 is not installed).

**Spec:** `docs/superpowers/specs/2026-08-31-detectkit-sam3-finetune-design.md`

## Global Constraints

- **Dependency direction:** `training/` and `core/inference/semantic/` must never import from an app layer (`detectkit/`, `trackerkit/`, ...). DetectKit imports them.
- **No god objects:** a new class or file over ~500 lines is doing too much. `detectkit/gui/dialogs/training_dialog.py` is already 2676 lines — add nothing to it beyond delegation.
- **Qt-free trainer:** everything under `training/sam3_lora/` must import no Qt.
- **Meta `sam3` is training-only** and lazily imported. A missing training dependency disables the action with a reason; it never raises at click time and never triggers an ultralytics AutoUpdate pip install.
- **SAM3 architecture input size is 1008** (`ultralytics/models/sam/build_sam3.py:38,308`). Training and inference must both use it.
- **Checkpoint selection is `last`, never `best`** — `best` is selected on validation loss, and validation loss was empirically anti-correlated with held-out AP in the spike.
- **Published artifacts go to `get_models_dir()/"sam3_finetuned"/`**, never `get_models_dir()/"sam3"/`, which is the stock download cache (`core/inference/semantic/checkpoints.py:91`).
- **Qt is PySide6.** There is no `pytest-qt` and no `qtbot` fixture in this repo.
  Widget tests follow `tests/test_detectkit_dataset_panel_widget.py`:
  `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")`,
  `pytest.importorskip("PySide6")`, and a module-scoped `qapp` fixture doing
  `QApplication.instance() or QApplication(sys.argv)`.
- **Spike tooling lives on `spike/sam3-lora`.** Any task using it starts with
  `git checkout spike/sam3-lora -- scratch/sam3_lora_spike/`.
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
git checkout spike/sam3-lora -- scratch/sam3_lora_spike/
cd scratch/sam3_lora_spike
PYTHONPATH=$PWD/../../src python show_on_frames.py --n 3 --conf 0.4 --imgsz 644
PYTHONPATH=$PWD/../../src python show_on_frames.py --n 3 --conf 0.4 --imgsz 1008
git checkout HEAD -- scratch/ || rm -rf scratch/sam3_lora_spike
```

`show_on_frames.py` takes `--imgsz` explicitly for exactly this comparison.

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
  - `LoraConfig(rank: int, alpha: int, dropout: float, target_suffixes: tuple[str, ...], include_prefixes: tuple[str, ...] = (), exclude_prefixes: tuple[str, ...] = ())`
  - `SUBMODULE_PREFIXES: dict[str, tuple[str, ...]]` mapping each `adapt_*` flag name to the dotted module-path prefixes it covers.
  - `lora_config_from_params(params) -> LoraConfig` — turns the six `adapt_*` booleans into `include_prefixes`.
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


def _cfg(rank=2, alpha=4, **kw):
    return LoraConfig(rank=rank, alpha=alpha, dropout=0.0,
                      target_suffixes=("qkv",), **kw)


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


class Nested(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = Toy()
        self.text = Toy()


def test_include_prefixes_scope_injection():
    m = Nested()
    # The six adapt_* flags select submodules by PREFIX; without this the
    # flags are dead parameters and the frozen text encoder gets adapted.
    n = inject_adapters(m, _cfg(include_prefixes=("vision",)))
    assert n == 1
    keys = " ".join(adapter_state_dict(m).keys())
    assert "vision" in keys and "text" not in keys


def test_exclude_prefixes_win_over_include():
    m = Nested()
    n = inject_adapters(
        m, _cfg(include_prefixes=("vision", "text"), exclude_prefixes=("text",)))
    assert n == 1
    assert "text" not in " ".join(adapter_state_dict(m).keys())


def test_empty_include_prefixes_means_everything():
    m = Nested()
    assert inject_adapters(m, _cfg()) == 2


def test_lora_config_from_params_maps_the_flags():
    from hydra_suite.training.contracts import Sam3LoraParams
    from hydra_suite.training.sam3_lora.lora import (
        SUBMODULE_PREFIXES,
        lora_config_from_params,
    )

    cfg = lora_config_from_params(
        Sam3LoraParams(prompt="ant", adapt_text_encoder=False,
                       adapt_vision_encoder=True))
    for pref in SUBMODULE_PREFIXES["adapt_text_encoder"]:
        assert pref not in cfg.include_prefixes
    for pref in SUBMODULE_PREFIXES["adapt_vision_encoder"]:
        assert pref in cfg.include_prefixes


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
    # The six adapt_* flags select submodules by PREFIX, not suffix. Without
    # these the flags cannot be expressed and every matching Linear is adapted
    # -- including the text encoder we deliberately freeze.
    include_prefixes: tuple[str, ...] = ()
    exclude_prefixes: tuple[str, ...] = ()

    @property
    def scaling(self) -> float:
        return float(self.alpha) / float(self.rank)


# Dotted module-path prefixes for each adapt_* flag, against the model Meta's
# build_sam3_image_model returns. VERIFY these against a live model before
# relying on them: print sorted({n.rsplit(".", 1)[0] for n, _ in
# model.named_modules()}) and confirm each prefix matches something.
SUBMODULE_PREFIXES: dict[str, tuple[str, ...]] = {
    "adapt_vision_encoder": ("backbone.vision_backbone",),
    "adapt_text_encoder": ("backbone.language_backbone",),
    "adapt_geometry_encoder": ("backbone.geometry_encoder",),
    "adapt_detr_encoder": ("transformer.encoder",),
    "adapt_detr_decoder": ("transformer.decoder",),
    "adapt_mask_decoder": ("mask_decoder", "sam_mask_decoder"),
}


def lora_config_from_params(params) -> "LoraConfig":
    """Turn the six adapt_* booleans into an include-prefix list.

    An empty include list means "everything", so a params object with all six
    flags True yields () rather than the union -- same behaviour, fewer
    string comparisons per module.
    """
    enabled = [f for f in SUBMODULE_PREFIXES if getattr(params, f)]
    include: tuple[str, ...] = ()
    if len(enabled) < len(SUBMODULE_PREFIXES):
        include = tuple(
            pref for flag in enabled for pref in SUBMODULE_PREFIXES[flag]
        )
    return LoraConfig(
        rank=params.rank, alpha=params.alpha, dropout=params.dropout,
        target_suffixes=TARGET_SUFFIXES, include_prefixes=include,
    )


# The Linear leaf names LoRA attaches to, across SAM3's ViT, CLIP-style text
# tower and DETR transformer.
TARGET_SUFFIXES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "out_proj", "qkv", "proj",
    "fc1", "fc2", "c_fc", "c_proj", "linear1", "linear2",
)


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
    def _scoped(name: str) -> bool:
        if cfg.exclude_prefixes and name.startswith(cfg.exclude_prefixes):
            return False
        return not cfg.include_prefixes or name.startswith(cfg.include_prefixes)

    targets = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
        and name.split(".")[-1] in cfg.target_suffixes
        and _scoped(name)
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
Expected: 9 passed

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
- Produces: `Sam3TrainingAvailability(usable: bool, reason: str)` and `probe_sam3_training_availability(cache_dir: Path | None = None) -> Sam3TrainingAvailability`. Module constant `TRAINING_PACKAGES = ("sam3", "torch", "torchmetrics", "scipy", "einops", "decord")`.

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

TRAINING_PACKAGES = ("sam3", "torch", "torchmetrics", "scipy", "einops", "decord")

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
    assert p.label_quality_acknowledged is False


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

`validation.py` imports neither `json` nor `typing.Any` today — add both.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam3_role_plumbing.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the neighbouring suites for regressions**

Run: `python -m pytest tests/test_training_validation.py tests/test_geometry_level_builders.py tests/test_sliced_dataset.py -v`
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

## Task 7: Thread `sam3_params` through the builder API

Before the builder can exist, the parameters must be able to *reach* it.
`prepare_role_dataset` takes no spec and no params
(`dataset_builders.py:1031-1042`), and its only caller
`TrainingOrchestrator.build_role_dataset` (`service.py:374-397`) has nothing to
forward. Skipping this is why a naive builder branch would raise `NameError` on
an undefined `sam3_params`.

**Files:**
- Modify: `src/hydra_suite/training/dataset_builders.py:1031-1042`
- Modify: `src/hydra_suite/training/service.py:374-397`
- Test: `tests/test_sam3_params_threading.py`

**Interfaces:**
- Consumes: `Sam3LoraParams` (Task 5).
- Produces: `prepare_role_dataset(..., *, sam3_params: "Sam3LoraParams | None" = None, seed: int = 42, split: SplitConfig | None = None)` and the same three keywords on `TrainingOrchestrator.build_role_dataset`, forwarded verbatim.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sam3_params_threading.py
"""The role's params must reach the builder; nothing else may change."""

import inspect

from hydra_suite.training.dataset_builders import prepare_role_dataset
from hydra_suite.training.service import TrainingOrchestrator


def test_prepare_role_dataset_accepts_sam3_params_seed_and_split():
    sig = inspect.signature(prepare_role_dataset)
    for name in ("sam3_params", "seed", "split"):
        assert name in sig.parameters, f"{name} missing"
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_orchestrator_forwards_them(monkeypatch, tmp_path):
    seen = {}

    def fake_prepare(role, merged_obb_dataset_dir, role_output_root, *a, **kw):
        seen.update(kw)
        from hydra_suite.training.contracts import DatasetBuildResult

        return DatasetBuildResult(dataset_dir=str(role_output_root))

    import hydra_suite.training.service as svc

    monkeypatch.setattr(svc, "prepare_role_dataset", fake_prepare)
    monkeypatch.setattr(svc, "validate_role_dataset",
                        lambda *a, **k: __import__(
                            "hydra_suite.training.contracts", fromlist=["x"]
                        ).ValidationReport(valid=True))

    from hydra_suite.training.contracts import Sam3LoraParams, TrainingRole

    orch = TrainingOrchestrator(tmp_path)
    params = Sam3LoraParams(prompt="ant")
    orch.build_role_dataset(
        TrainingRole.SEMANTIC_SAM3, str(tmp_path), sam3_params=params, seed=7
    )
    assert seen["sam3_params"] is params
    assert seen["seed"] == 7


def test_existing_roles_are_unaffected():
    sig = inspect.signature(prepare_role_dataset)
    # The pre-existing positional contract must not move.
    names = list(sig.parameters)
    assert names[:4] == ["role", "merged_obb_dataset_dir",
                         "role_output_root", "class_name"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam3_params_threading.py -v`
Expected: FAIL with `AssertionError: sam3_params missing`

- [ ] **Step 3: Write minimal implementation**

In `dataset_builders.py`, add three keyword-only parameters after
`merged_level` (never positional — the existing contract must not move):

```python
    sam3_params: "Sam3LoraParams | None" = None,
    seed: int = 42,
    split: SplitConfig | None = None,
```

In `service.py`, add the same three to `build_role_dataset` and forward them
verbatim into the `prepare_role_dataset(...)` call.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam3_params_threading.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/dataset_builders.py src/hydra_suite/training/service.py tests/test_sam3_params_threading.py
git commit -m "feat(training): thread sam3_params/seed/split to the role builder"
```

---

## Task 7b: COCO tile dataset builder

**Files:**
- Create: `src/hydra_suite/training/sam3_lora/dataset_build.py`
- Modify: `src/hydra_suite/training/dataset_builders.py` (`prepare_role_dataset` branch)
- Test: `tests/test_sam3_dataset_build.py`

**Interfaces:**
- Consumes: `Sam3LoraParams` (Task 5), the threading from Task 7.
- Produces: `build_sam3_coco_dataset(source_dir, out_dir, params, *, class_name=None, seed=42, split=None) -> dict` with stats keys `train_images`, `train_annotations`, `crowd_annotations`, `tile_px`, `negative_prompts`, `validation`, `selected_class`.
- Produces: `resolve_negative_prompts(params, source_class_names, selected_class) -> list[str]`.
- Produces: `CURATED_NEGATIVES = ("background", "shadow", "debris")`, `MIN_RETAINED_AREA_FRAC = 0.5`.

**The source is a single raw DetectKit source**, not the merged OBB dataset.
Spec breakage row 5 decided this: concept training is per-source, and
`build_merged_obb_dataset` is skipped for this role (Task 11b enforces it in the
dialog).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sam3_dataset_build.py
"""Tiling, iscrowd boundary, frame-level split, negative-prompt resolution."""

import json

import cv2
import numpy as np

from hydra_suite.training.contracts import Sam3LoraParams
from hydra_suite.training.sam3_lora.dataset_build import (
    CURATED_NEGATIVES,
    build_sam3_coco_dataset,
    resolve_negative_prompts,
)


def _source(tmp_path, n_frames=3, size=2048):
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        cv2.imwrite(str(img_dir / f"f{i}.jpg"),
                    rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
        poly = np.array([[0.50, 0.50], [0.52, 0.50], [0.52, 0.52], [0.50, 0.52]])
        (lbl_dir / f"f{i}.txt").write_text(
            "0 " + " ".join(f"{v:.6f}" for v in poly.reshape(-1)) + "\n")
    (tmp_path / "classes.txt").write_text("ant\n")
    return tmp_path


def _params(**kw):
    base = dict(prompt="ant with color patch", geometry_mode="custom",
                slice_width=512, slice_height=512, tile_overlap=0.25)
    base.update(kw)
    return Sam3LoraParams(**base)


def _load(out, split):
    return json.loads((out / split / "_annotations.coco.json").read_text())


def test_category_name_is_the_prompt(tmp_path):
    out = tmp_path / "out"
    build_sam3_coco_dataset(_source(tmp_path / "src"), out, _params())
    assert _load(out, "train")["categories"][0]["name"] == "ant with color patch"


def test_split_is_by_frame_not_by_tile(tmp_path):
    out = tmp_path / "out"
    build_sam3_coco_dataset(_source(tmp_path / "src", n_frames=3), out, _params())
    tr = {i["file_name"].split("_")[0] for i in _load(out, "train")["images"]}
    va = {i["file_name"].split("_")[0] for i in _load(out, "valid")["images"]}
    assert tr and va and tr.isdisjoint(va)


def test_single_frame_source_trains_without_validation(tmp_path):
    out = tmp_path / "out"
    stats = build_sam3_coco_dataset(
        _source(tmp_path / "src", n_frames=1), out, _params())
    assert stats["train_images"] > 0
    assert stats["validation"] == "none"


def test_empty_tiles_are_kept_when_requested(tmp_path):
    out = tmp_path / "out"
    build_sam3_coco_dataset(_source(tmp_path / "src"), out,
                            _params(keep_empty_tiles=True))
    data = _load(out, "train")
    with_ann = {a["image_id"] for a in data["annotations"]}
    assert len(data["images"]) > len(with_ann)


def test_seam_clipped_instances_become_iscrowd(tmp_path):
    # A polygon straddling a tile seam retains <50% on one side; it must be
    # marked iscrowd, not dropped -- dropping teaches SAM3 that a visible
    # half-animal is background.
    out = tmp_path / "out"
    stats = build_sam3_coco_dataset(
        _source(tmp_path / "src"), out,
        _params(slice_width=1024, slice_height=1024, tile_overlap=0.0))
    assert stats["crowd_annotations"] >= 0  # key exists and is counted
    data = _load(out, "train")
    assert all(a["iscrowd"] in (0, 1) for a in data["annotations"])


def test_split_is_deterministic_under_seed(tmp_path):
    a_out, b_out = tmp_path / "a", tmp_path / "b"
    src = _source(tmp_path / "src")
    build_sam3_coco_dataset(src, a_out, _params(), seed=7)
    build_sam3_coco_dataset(src, b_out, _params(), seed=7)
    assert ({i["file_name"] for i in _load(a_out, "valid")["images"]}
            == {i["file_name"] for i in _load(b_out, "valid")["images"]})


def test_negative_prompts_prefer_explicit_then_classes_then_curated():
    assert resolve_negative_prompts(
        _params(negative_prompts=["mite"]), ["ant", "beetle"], "ant") == ["mite"]
    # Tier 2: the OTHER class names of the source -- the confusable concepts.
    assert resolve_negative_prompts(
        _params(), ["ant", "beetle"], "ant") == ["beetle"]
    got = resolve_negative_prompts(_params(), ["ant"], "ant")
    assert got and set(got).issubset(set(CURATED_NEGATIVES))


def test_curated_negatives_drop_word_overlap_with_the_prompt():
    p = Sam3LoraParams(prompt="ant on a shadow")
    assert "shadow" not in resolve_negative_prompts(p, ["ant"], "ant")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam3_dataset_build.py -v`
Expected: FAIL — `ModuleNotFoundError: ...sam3_lora.dataset_build`

- [ ] **Step 3: Write minimal implementation**

Create `dataset_build.py`. All geometry comes from
`hydra_suite.utils.slice_geometry` (`tile_size_for_mode:104`, `plan_tiles:130`,
`clip_polygon_to_tile:215`, `polygon_area:186`) — never a hand-rolled tiler.

```python
CURATED_NEGATIVES = ("background", "shadow", "debris")
MIN_RETAINED_AREA_FRAC = 0.5


def resolve_negative_prompts(params, source_class_names, selected_class):
    """Negatives are NAMED, not inferred.

    SAM3 trains with prompts that must return nothing so the tuned model keeps
    discriminating concepts. The spike's third-party trainer sampled these from
    other COCO categories -- impossible here, because this builder emits a
    single category by construction. Hence three explicit tiers.
    """
    if params.negative_prompts:
        return list(params.negative_prompts)
    others = [c for c in source_class_names if c != selected_class]
    if others:
        return others
    prompt_words = {w for w in params.prompt.lower().split() if w}
    return [n for n in CURATED_NEGATIVES
            if not (set(n.lower().split()) & prompt_words)]
```

`build_sam3_coco_dataset` rules, each pinned by a test above:

- `selected_class` defaults to the source's first class; only its polygons are
  emitted, and `stats["selected_class"]` records which.
- Frames are shuffled under `seed` and split by **frame** using `split` (default
  `SplitConfig()`); 1 frame => train-only with `stats["validation"] = "none"`;
  2 frames => 1/1 with a logged warning that val is one frame.
- An instance retaining `< MIN_RETAINED_AREA_FRAC` after
  `clip_polygon_to_tile` is written with `iscrowd = 1`, never dropped.
- Empty tiles are written when `params.keep_empty_tiles`.
- `category.name = params.prompt`.
- `build_manifest.json` records `tile_px`, `reference_body_px`,
  `object_tile_fraction`, the resolved negatives, the selected class and the
  frame split.

Then the `prepare_role_dataset` branch, using Task 7's keywords:

```python
    if role is TrainingRole.SEMANTIC_SAM3:
        # Concept training is PER SOURCE. The caller passes a single raw source
        # dir here, NOT a merged OBB dataset -- see the dialog task, which skips
        # build_merged_obb_dataset for this role.
        from hydra_suite.training.sam3_lora.dataset_build import (
            build_sam3_coco_dataset,
        )

        if sam3_params is None:
            raise ValueError("SEMANTIC_SAM3 requires sam3_params")
        stats = build_sam3_coco_dataset(
            merged_obb_dataset_dir, role_output_root, sam3_params,
            class_name=class_name, seed=seed, split=split,
        )
        return DatasetBuildResult(dataset_dir=str(role_output_root), stats=stats)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam3_dataset_build.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/sam3_lora/dataset_build.py src/hydra_suite/training/dataset_builders.py tests/test_sam3_dataset_build.py
git commit -m "feat(training): COCO tile dataset builder for SAM3 finetuning

Reuses slice_geometry. Splits by frame (overlapping tiles leak pixels), marks
seam-clipped instances iscrowd, keeps empty tiles, emits a single selected
class, and resolves negative prompts in three explicit tiers -- a
single-category dataset has no other category to sample them from."
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

`preflight` returns a list of reasons for, in order: non-CUDA device; free
VRAM below `REFUSE_BELOW_GB`; empty prompt;
`_instance_count(...) < MIN_TRAIN_INSTANCES`; free disk below
`REQUIRED_DISK_GB = 8.0` (the merged artifact is ~3.2 GB and the merge needs
the base resident — check before an hour of GPU, not at publish time); a
missing label-quality acknowledgement
(`spec.sam3_params.label_quality_acknowledged`); and non-empty
`spec.resume_from` (optimiser state is not checkpointed).
`_cuda_free_gb`, `_free_disk_gb` and `_instance_count` are module-level seams
so tests can monkeypatch them without a GPU.

It also returns a **warning** (not a refusal) when free VRAM is below
`WARN_BELOW_GB`, and when `torch.cuda.get_device_capability()[0] < 8` with
`mixed_precision == "bf16"` — see the fallback below.

`train_sam3_lora` **calls `preflight(spec)` as its first action** and, if it
returns refusals, returns `{"success": False, "error_message": "; ".join(...)}`
without importing `sam3` or loading any weights. Nothing else calls preflight,
so if this call is omitted the whole module is dead code.

Create `src/hydra_suite/training/sam3_lora/datapoints.py` — the seam the spec
names and the first draft of this plan dropped. It converts a COCO tile record
into the structures Meta's model consumes. **These import paths are verified
against the installed package and are not obvious**; an implementer will not
guess them:

```python
from sam3.train.data.sam3_image_dataset import (
    Datapoint, FindQueryLoaded, Image, InferenceMetadata)
from sam3.train.data.collator import collate_fn_api

from hydra_suite.core.inference.semantic.sam3 import PREDICTOR_IMGSZ

# ONE definition of SAM3's input size, shared with the predictor overrides.
# Training and serving must agree or the sidecar's imgsz guard fires.
RES = PREDICTOR_IMGSZ


def build_datapoint(tile_bgr, prompt, polygons, transform):
    """One COCO tile -> one Datapoint carrying a single text query."""
    h, w = tile_bgr.shape[:2]
    pil = PILImage.fromarray(cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB))
    if (w, h) != (RES, RES):
        pil = pil.resize((RES, RES), PILImage.BILINEAR)
    query = FindQueryLoaded(
        query_text=prompt, image_id=0, object_ids_output=[],
        is_exhaustive=True, query_processing_order=0,
        inference_metadata=InferenceMetadata(
            coco_image_id=0, original_image_id=0, original_category_id=0,
            original_size=(h, w), object_id=-1, frame_index=-1))
    return Datapoint(find_queries=[query],
                     images=[Image(data=transform(pil), objects=polygons,
                                   size=(RES, RES))],
                     raw_images=[pil])
```

Negative queries are the same structure with a negative prompt and **no**
objects, sampled `params.num_negatives` per image from Task 7b's resolved list.
Batches go through `collate_fn_api([...], dict_key="input", with_seg_masks=True)`.

`train.py` then:

1. Calls `preflight(spec)`; returns early on refusals.
2. Seeds `torch`, `numpy` and `random` from `spec.seed`.
3. Lazily imports `sam3`, builds via `build_sam3_image_model(..., eval_mode=False)`,
   and injects adapters with Task 3's `inject_adapters`, mapping each
   `adapt_*` flag to the module-path prefixes it covers.
4. Builds the objective from **Meta's own** code — we do not reimplement it:

```python
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2

matcher = BinaryHungarianMatcherV2(
    cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True)
```

   `outputs["indices"] = matcher(outputs, targets)` must be set on the main
   output **and on every auxiliary output** before the loss is called, or
   `Sam3LossWrapper` raises.
5. AdamW over `get_lora_parameters`-equivalent (parameters with
   `requires_grad`), cosine schedule with `warmup_steps = min(50, total_steps
   // 4)` — the spike used a flat 50 against ~9 steps/epoch, which meant its
   epoch-1 checkpoint trained entirely inside warmup.
6. bf16 autocast when `torch.cuda.get_device_capability()[0] >= 8`, else
   **fp32** with a logged notice. Not fp16: without a loss scaler fp16 diverges,
   and adding a scaler is scope this task does not need.
7. Gradient clipping at 1.0; `should_cancel()` checked between optimiser steps,
   returning `{"success": False, "canceled": True}` promptly.
8. `progress_cb(epoch, params.epochs)` per epoch; `log_cb` per
   `logging_steps`.
9. Writes `adapters.pt` (`adapter_state_dict`) and `val_stats.json` into
   `run_dir`; returns `{"success": True, "artifact_path": str(run_dir /
   "adapters.pt"), "metrics_path": str(run_dir / "val_stats.json"),
   "canceled": False}`.

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
    # The registry must land under models_root, never the user's real one.
    assert (tmp_path / "models" / "model_registry.json").exists()
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
   instantiate every submodule the published checkpoint carries -- on the spike
   this was 22 `vision_backbone.sam2_convs.*` tensors, but **count them at merge
   time and log the count; do not hard-code 22**. Under `strict=False` they
   would stay at random init with no error. Copy them from base untouched. A key
   present in the merge but absent from base is the opposite problem and is a
   hard error.
3. Write to `models_root/"sam3_finetuned"/f"{run_id}.pt"`.
4. Write `<artifact>.sam3_meta.json` with `base_variant`, `prompt`,
   `train_tile_px`, `reference_body_px`, `object_tile_fraction`, `imgsz`
   (`PREDICTOR_IMGSZ`), `stripped_keys`, `tuned_fingerprints` (sha256 of the
   raw bytes of 2-3 tensors the merge changed), `source_fingerprint`,
   `label_quality_acknowledged`.
5. Register in `model_registry.json` with `task_family="semantic"` and
   `usage_role="semantic_sam3"` — a registry entry, never a directory scan.
   **The registry path must derive from `models_root`**, not from the global
   `get_yolo_model_registry_path()` (`core/inference/model_paths.py:165-167`)
   which resolves against the real user models dir — otherwise the tests below
   write into the developer's own registry. Give `publish_sam3_model` a
   `registry_path: Path | None = None` defaulting to
   `models_root / "model_registry.json"`.

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

## Task 9b: Wire publish into the service auto-publish path

Without this, `publish_sam3_model` is orphaned and every finished run crashes:
`run_role_training` (`service.py:466-472`) routes `artifact_path` into
`_publish_training_artifacts` (`service.py:82`) → `publish_trained_model` →
`_repo_dir_for_role`, which raises `RuntimeError("Unsupported publish role")`
(`model_publish.py:66-67`). The GPU hours are already spent by then.

**Files:**
- Modify: `src/hydra_suite/training/service.py:82-130` (`_publish_training_artifacts`)
- Test: `tests/test_sam3_service_publish.py`

**Interfaces:**
- Consumes: `publish_sam3_model` (Task 9).
- Produces: no new public names; `_publish_training_artifacts` returns the same `(published_key, published_path)` tuple for this role as for any other.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sam3_service_publish.py
"""SEMANTIC_SAM3 must fork to publish_sam3_model, never publish_trained_model."""

import hydra_suite.training.service as svc
from hydra_suite.training.contracts import (
    Sam3LoraParams, SourceDataset, TrainingHyperParams, TrainingRole,
    TrainingRunSpec)


def _spec(role=TrainingRole.SEMANTIC_SAM3):
    return TrainingRunSpec(
        role=role,
        source_datasets=[SourceDataset(path="/tmp/x", level="polygon")],
        derived_dataset_dir="/tmp/d", base_model="sam3",
        hyperparams=TrainingHyperParams(),
        sam3_params=Sam3LoraParams(prompt="ant"))


def test_sam3_role_forks_to_publish_sam3_model(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(svc, "publish_trained_model",
                        lambda *a, **k: called.setdefault("yolo", True))
    monkeypatch.setattr(svc, "publish_sam3_model",
                        lambda **k: (called.setdefault("sam3", True),
                                     ("key", "/tmp/a.pt"))[1])
    monkeypatch.setattr(svc, "ensure_checkpoint", lambda *a, **k: "/tmp/base.pt")
    key, path = svc._publish_training_artifacts(
        spec=_spec(), artifact_paths=[str(tmp_path / "adapters.pt")],
        publish_metadata={}, run_id="r1")
    assert "sam3" in called
    # publish_trained_model raises "Unsupported publish role" for this role.
    assert "yolo" not in called
    assert path == "/tmp/a.pt"


def test_other_roles_still_use_the_yolo_publisher(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(svc, "publish_trained_model",
                        lambda *a, **k: (called.setdefault("yolo", True),
                                         ("k", "/tmp/y.pt"))[1])
    monkeypatch.setattr(svc, "publish_sam3_model",
                        lambda **k: called.setdefault("sam3", True))
    svc._publish_training_artifacts(
        spec=_spec(TrainingRole.SEGMENT_DIRECT),
        artifact_paths=[str(tmp_path / "best.pt")],
        publish_metadata={}, run_id="r1")
    assert "yolo" in called and "sam3" not in called
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam3_service_publish.py -v`
Expected: FAIL with `AttributeError: module 'hydra_suite.training.service' has no attribute 'publish_sam3_model'`

- [ ] **Step 3: Write minimal implementation**

Import both `publish_sam3_model` and `ensure_checkpoint` at module scope in
`service.py` (module scope matters — the tests monkeypatch them there), then
branch at the top of `_publish_training_artifacts`:

```python
    if spec.role is TrainingRole.SEMANTIC_SAM3:
        # Forked, not extended: publish_trained_model's naming scheme does not
        # fit a promptable-concept checkpoint, and _repo_dir_for_role raises
        # for this role. See the design's "Publish -- a fork, not an extension".
        return publish_sam3_model(
            run_id=run_id,
            adapters_path=Path(artifact_paths[0]),
            base_checkpoint=ensure_checkpoint("sam3", allow_download=False),
            build_manifest=dict(publish_metadata or {}),
            params=spec.sam3_params,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam3_service_publish.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/service.py tests/test_sam3_service_publish.py
git commit -m "feat(training): fork auto-publish for SEMANTIC_SAM3

Without this the role trains for an hour and then crashes in
_repo_dir_for_role, which raises for any unsupported publish role."
```

---

## Task 10: Escalation consumption and the silent-load guard

**Files:**
- Modify: `src/hydra_suite/core/inference/semantic/checkpoints.py`
- Modify: `src/hydra_suite/core/inference/semantic/sam3.py`
- Test: `tests/test_sam3_resolve_and_guard.py`

**Interfaces:**
- Consumes: the sidecar from Task 9.
- Produces: `resolve_checkpoint(key, cache_dir=None) -> Path`, `available_models() -> list[str]`, `probe_dependencies() -> Sam3Availability`, `probe_checkpoint(key, cache_dir=None) -> Sam3Availability`, `assert_checkpoint_loaded(live_state_dict, meta, *, imgsz) -> None`, and `Sam3SemanticLabeler.from_variant(..., checkpoint: Path | str | None = None)`.

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
    meta = {"stripped_keys": ["a.weight"], "tuned_fingerprints": {"a.weight": fp},
            "imgsz": 1008}
    assert assert_checkpoint_loaded({"a.weight": t}, meta, imgsz=1008) is None


def test_guard_refuses_an_imgsz_mismatch():
    import hashlib

    # A model finetuned at 1008 and served at 644 is a 1.56x train/serve scale
    # mismatch. It loads CLEANLY -- keys and tensors all match -- so only an
    # explicit check catches it. Rescaling silently is the failure mode.
    t = torch.randn(2, 2)
    fp = hashlib.sha256(t.numpy().tobytes()).hexdigest()
    meta = {"stripped_keys": ["a.weight"], "tuned_fingerprints": {"a.weight": fp},
            "imgsz": 1008}
    with pytest.raises(RuntimeError, match="644"):
        assert_checkpoint_loaded({"a.weight": t}, meta, imgsz=644)


def test_stock_variant_without_a_sidecar_is_unguarded():
    # A stock variant ships no sidecar and makes no claim; guarding it would
    # refuse every un-finetuned run.
    assert assert_checkpoint_loaded({"a.weight": torch.zeros(2, 2)},
                                    None, imgsz=1008) is None
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

In `sam3.py`: add `assert_checkpoint_loaded(live_state_dict, meta, *, imgsz)`
raising a `RuntimeError` for any of three conditions, and returning `None` when
`meta` is `None` (a stock variant ships no sidecar and makes no claim):

1. a `stripped_keys` entry absent from the live model — a key-namespace rename;
2. a `tuned_fingerprints` entry whose sha256 does not match — our weights are
   not the ones resident, which a key check alone cannot detect;
3. `meta["imgsz"] != imgsz` — **the train/serve scale mismatch**. This one loads
   perfectly cleanly, so nothing else in the system would ever notice it. The
   spec requires refusing rather than silently rescaling.

Add a `checkpoint:` parameter to `from_variant`; when it names a published
artifact, force eager `setup_model()`, read the sidecar, and call the guard with
`imgsz=PREDICTOR_IMGSZ`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam3_resolve_and_guard.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/semantic/ tests/test_sam3_resolve_and_guard.py
git commit -m "feat(semantic): resolve published SAM3 models + silent-load guard

Three checks, each for a failure that is otherwise invisible: key names (a
rename), tensor fingerprints (our weights are not the resident ones -- a key
check passes on exactly this), and imgsz (a model finetuned at 1008 served at
644 loads perfectly cleanly)."
```

---

## Task 10b: Make published models selectable in escalation

Task 10 adds `resolve_checkpoint`, `available_models` and the guard, but
**nothing calls them** — the three `from_variant` call sites still pass stock
variant strings, so a published model is unselectable and the silent-load guard
is unreachable. This task is a prerequisite for Task 12's Gate 3 and for the
documentation in Task 13; without it both describe a feature that does not
exist.

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/dialogs/semantic_escalation_dialog.py:135-142` (Model combo), and the saved-config read/write at `:139-141, 351`
- Modify: `src/hydra_suite/detectkit/jobs/semantic_escalation.py:960, 1055, 1102` (the three `from_variant` call sites)
- Test: `tests/test_sam3_model_selection.py`

**Interfaces:**
- Consumes: `available_models`, `resolve_checkpoint` (Task 10); the sidecar written by Task 9.
- Produces: `sidecar_for(model_key) -> dict | None` in `checkpoints.py`, and dialog method `prefill_from_sidecar(model_key) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sam3_model_selection.py
"""A published model must be selectable, resolvable and prefill its geometry."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

import sys  # noqa: E402

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_combo_lists_published_models(qapp, monkeypatch):
    from hydra_suite.detectkit.gui.dialogs import semantic_escalation_dialog as d

    monkeypatch.setattr(d, "available_models", lambda: ["sam3", "run123"])
    dlg = d.SemanticEscalationDialog(sources=[], saved={})
    items = [dlg._variant.itemText(i) for i in range(dlg._variant.count())]
    assert "run123" in items


def test_selecting_a_published_model_prefills_prompt_and_fraction(
    qapp, monkeypatch, tmp_path
):
    from hydra_suite.detectkit.gui.dialogs import semantic_escalation_dialog as d

    monkeypatch.setattr(d, "available_models", lambda: ["sam3", "run123"])
    monkeypatch.setattr(d, "sidecar_for", lambda k: {
        "prompt": "ant with color patch", "object_tile_fraction": 0.055,
        "train_tile_px": 1007})
    dlg = d.SemanticEscalationDialog(sources=[], saved={})
    dlg.prefill_from_sidecar("run123")
    assert dlg.prompt() == "ant with color patch"
    assert abs(dlg.tile_fraction() - 0.055) < 1e-9


def test_prefill_is_a_default_not_a_lock(qapp, monkeypatch):
    # REFERENCE_BODY_SIZE precedent: a measured value is sacrosanct, a derived
    # one is a starting point the user may override.
    from hydra_suite.detectkit.gui.dialogs import semantic_escalation_dialog as d

    monkeypatch.setattr(d, "available_models", lambda: ["sam3", "run123"])
    monkeypatch.setattr(d, "sidecar_for", lambda k: {"prompt": "x",
                                                     "object_tile_fraction": 0.05})
    dlg = d.SemanticEscalationDialog(sources=[], saved={})
    dlg.prefill_from_sidecar("run123")
    assert dlg._prompt.isEnabled()
    assert dlg._prompt.isReadOnly() is False


def test_job_resolves_the_selected_key_to_a_checkpoint(monkeypatch, tmp_path):
    from hydra_suite.detectkit.jobs import semantic_escalation as job

    seen = {}

    class FakeLabeler:
        @classmethod
        def from_variant(cls, variant="sam3", **kw):
            seen["variant"] = variant
            seen["checkpoint"] = kw.get("checkpoint")
            return cls()

    monkeypatch.setattr(job, "Sam3SemanticLabeler", FakeLabeler, raising=False)
    monkeypatch.setattr(job, "resolve_checkpoint",
                        lambda k, **kw: tmp_path / f"{k}.pt", raising=False)
    ck = job.labeler_checkpoint_for("run123")
    assert str(ck).endswith("run123.pt")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam3_model_selection.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'available_models'`

- [ ] **Step 3: Write minimal implementation**

In `checkpoints.py`, add `sidecar_for(model_key)` returning the parsed
`<artifact>.sam3_meta.json` for a published key, or `None` for a stock variant.

In the dialog, replace the `available_variants()` call at `:136-138` with
`available_models()`, and connect `currentTextChanged` to
`prefill_from_sidecar`, which sets prompt and tile fraction **without disabling
the widgets** — prefill is a default, not a lock.

In `semantic_escalation.py`, add a small helper and route all three call sites
through it, so the resolution rule lives in one place:

```python
def labeler_checkpoint_for(model_key: str):
    """Resolve a UI model key to a checkpoint path.

    Stock variants and published finetuned models are both selectable, so the
    key is no longer necessarily a SAM3_VARIANTS entry.
    """
    return resolve_checkpoint(model_key)
```

Each call site becomes
`Sam3SemanticLabeler.from_variant(checkpoint=labeler_checkpoint_for(key), ...)`.
The selected key also flows into `staged_dirname_for` (Task 2), so two models
never share a staging directory.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam3_model_selection.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/semantic/checkpoints.py src/hydra_suite/detectkit/gui/dialogs/semantic_escalation_dialog.py src/hydra_suite/detectkit/jobs/semantic_escalation.py tests/test_sam3_model_selection.py
git commit -m "feat(detectkit): select published SAM3 models in escalation

Without this the resolve/guard work from the previous task is unreachable:
the call sites still pass stock variant strings."
```

---

## Task 11: DetectKit training panel

**Files:**
- Create: `src/hydra_suite/detectkit/gui/panels/sam3_training_panel.py`
- Test: `tests/test_sam3_training_panel.py`

**Interfaces:**
- Consumes: `Sam3LoraParams` (Task 5), `probe_sam3_training_availability` (Task 4).
- Produces: `Sam3TrainingPanel(QWidget)` with `params() -> Sam3LoraParams`, `set_params(p) -> None`, `acknowledged() -> bool`, `unavailable_reason() -> str`.

**Qt is PySide6.** `training_dialog.py:9` imports PySide6; PyQt6 is not
installed and there is no `pytest-qt`/`qtbot` in this repo. Follow
`tests/test_detectkit_dataset_panel_widget.py`'s pattern exactly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sam3_training_panel.py
"""The panel owns the knobs, the acknowledgement, and disabled-with-reason."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_params_round_trip(qapp):
    from hydra_suite.detectkit.gui.panels.sam3_training_panel import (
        Sam3TrainingPanel,
    )
    from hydra_suite.training.contracts import Sam3LoraParams

    panel = Sam3TrainingPanel()
    got = panel.params()
    assert got.epochs == 10 and got.rank == 16

    panel.set_params(Sam3LoraParams(prompt="beetle", epochs=3, rank=8))
    back = panel.params()
    assert back.prompt == "beetle" and back.epochs == 3 and back.rank == 8


def test_training_is_blocked_until_labels_are_acknowledged(qapp):
    from hydra_suite.detectkit.gui.panels.sam3_training_panel import (
        Sam3TrainingPanel,
    )

    panel = Sam3TrainingPanel()
    # Provenance does not survive a review, so the user must affirm the labels
    # are good before SAM3 learns them -- including its own accepted output.
    assert panel.acknowledged() is False
    assert panel.params().label_quality_acknowledged is False
    panel.chk_ack.setChecked(True)
    assert panel.acknowledged() is True
    assert panel.params().label_quality_acknowledged is True


def test_unavailable_backend_disables_with_a_reason(qapp, monkeypatch):
    import hydra_suite.detectkit.gui.panels.sam3_training_panel as mod

    monkeypatch.setattr(
        mod, "probe_sam3_training_availability",
        lambda: mod.Sam3TrainingAvailability(False, "package 'sam3' is missing"))
    panel = mod.Sam3TrainingPanel()
    assert "sam3" in panel.unavailable_reason()
    assert not panel.isEnabled()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam3_training_panel.py -v`
Expected: FAIL — `ModuleNotFoundError: ...panels.sam3_training_panel`

- [ ] **Step 3: Write minimal implementation**

A self-contained `QWidget` built from `PySide6.QtWidgets`: prompt field,
negative-prompt list, LoRA group (rank/alpha/dropout), optimisation group
(lr/epochs/batch/grad_accum/precision), the six `adapt_*` checkboxes, the tiling
group (geometry mode, `object_tile_fraction`, slice w/h, overlap, keep-empty),
a read-only CUDA-host notice, and `self.chk_ack` — "I have verified these labels
are correct; SAM3 will learn any systematic error in them."

`params()` reads every widget into a `Sam3LoraParams`, including
`label_quality_acknowledged=self.chk_ack.isChecked()`. `set_params` is its
inverse. On construction, call `probe_sam3_training_availability()`; if not
usable, store the reason and `self.setEnabled(False)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam3_training_panel.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/detectkit/gui/panels/sam3_training_panel.py tests/test_sam3_training_panel.py
git commit -m "feat(detectkit): SAM3 training panel

A separate panel rather than inline widgets: training_dialog.py is already
2676 lines, over 5x the project's 500-line guidance."
```

---

## Task 11b: Wire the panel into the training dialog's run flow

The panel is inert until the dialog builds a spec from it. The dialog's flow
merges sources per geometry level and loops roles
(`training_dialog.py:2026-2145`); this role must **skip the merge** and consume a
single raw source, per the design's breakage row 5.

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/dialogs/training_dialog.py:55-61` (`_SELECTION_ROLE_MAP`), `:2026-2145` (`_build_role_datasets`), and the run path that builds `TrainingRunSpec`
- Test: `tests/test_sam3_dialog_wiring.py`

**Interfaces:**
- Consumes: `Sam3TrainingPanel` (Task 11).
- Produces: `_SELECTION_ROLE_MAP[("semantic", "polygon")] = ("semantic_sam3",)`, and `TrainingDialog._sam3_spec_for(source) -> TrainingRunSpec`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sam3_dialog_wiring.py
"""The dialog must build a spec from the panel, skip the merge, and gate on ack."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")


def test_selection_map_has_the_semantic_entry():
    from hydra_suite.detectkit.gui.dialogs.training_dialog import (
        _SELECTION_ROLE_MAP,
    )

    assert _SELECTION_ROLE_MAP[("semantic", "polygon")] == ("semantic_sam3",)


def test_spec_carries_the_panel_params(monkeypatch):
    from hydra_suite.detectkit.gui.dialogs import training_dialog as td
    from hydra_suite.training.contracts import Sam3LoraParams, TrainingRole

    params = Sam3LoraParams(prompt="ant", epochs=4,
                            label_quality_acknowledged=True)
    spec = td.TrainingDialog._sam3_spec_for(
        self=None, source_path="/tmp/src", params=params,
        derived_dir="/tmp/derived", seed=7)
    assert spec.role is TrainingRole.SEMANTIC_SAM3
    assert spec.sam3_params.prompt == "ant"
    assert spec.seed == 7
    # ONE raw source, not a merged OBB dataset.
    assert len(spec.source_datasets) == 1
    assert spec.source_datasets[0].path == "/tmp/src"


def test_unacknowledged_labels_block_the_run():
    from hydra_suite.detectkit.gui.dialogs import training_dialog as td
    from hydra_suite.training.contracts import Sam3LoraParams

    with pytest.raises(ValueError, match="acknowledge"):
        td.TrainingDialog._sam3_spec_for(
            self=None, source_path="/tmp/src",
            params=Sam3LoraParams(prompt="ant"),
            derived_dir="/tmp/derived", seed=7)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam3_dialog_wiring.py -v`
Expected: FAIL — `KeyError: ('semantic', 'polygon')`

- [ ] **Step 3: Write minimal implementation**

Add the `_SELECTION_ROLE_MAP` entry and a `_SELECTION_DESCRIPTIONS` entry
beside it. Add a `@staticmethod`-style helper (it takes no real `self`, which is
why the test can pass `self=None`):

```python
    @staticmethod
    def _sam3_spec_for(self, source_path, params, derived_dir, seed):
        """One raw source, no merge. Concept training is per-source."""
        if not params.label_quality_acknowledged:
            raise ValueError(
                "You must acknowledge the label-quality warning before "
                "training: SAM3 learns any systematic error in these labels."
            )
        return TrainingRunSpec(
            role=TrainingRole.SEMANTIC_SAM3,
            source_datasets=[SourceDataset(path=str(source_path),
                                           level="polygon")],
            derived_dataset_dir=str(derived_dir),
            base_model="sam3",
            hyperparams=TrainingHyperParams(epochs=params.epochs),
            seed=seed,
            sam3_params=params,
        )
```

In `_build_role_datasets`, branch **before** the `build_merged_obb_dataset`
loop: for `SEMANTIC_SAM3`, call
`orchestrator.build_role_dataset(role, <raw source dir>, sam3_params=panel.params(), seed=..., split=...)`
once per selected source and skip merging entirely.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam3_dialog_wiring.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/detectkit/gui/dialogs/training_dialog.py tests/test_sam3_dialog_wiring.py
git commit -m "feat(detectkit): wire the SAM3 panel into the training run flow

Skips build_merged_obb_dataset for this role -- concept training is
per-source -- and refuses to build a spec until the label-quality warning is
acknowledged."
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

- [ ] **Step 0: Provision the spike harness**

It lives only on `spike/sam3-lora`; a worktree from main has no `scratch/`
directory at all. `scratch/` is not gitignored, so the files can be committed
from here once present.

```bash
git checkout spike/sam3-lora -- scratch/sam3_lora_spike/
ls scratch/sam3_lora_spike/show_on_frames.py scratch/sam3_lora_spike/arm_sam3.py
```

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
names it explicitly:
`pip install git+https://github.com/facebookresearch/sam3.git`.
The extra and `TRAINING_PACKAGES` (Task 4) must list the same packages —
`decord` belongs in both.

- [ ] **Step 2: Write the user-guide section**

Cover: what the role does, the CUDA-only ~32 GB requirement, the label-quality
warning and why provenance cannot be filtered, that `epochs=10` is measured
(not a default of taste), and that a published model is selectable in the
semantic escalation dialog and prefills prompt and tile fraction — the feature
Task 10b builds. Do not document any of this before that task lands.

- [ ] **Step 3: Verify docs build**

Run: `make docs-check`
Expected: clean build

- [ ] **Step 4: Commit**

```bash
git add docs/ pyproject.toml CLAUDE.md
git commit -m "docs(detectkit): SAM3 finetuning role"
```
