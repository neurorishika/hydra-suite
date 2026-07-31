# ViTPose Fine-Tuning (PoseKit subprocess) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ViTPose fine-tuning to hydra as a first-class PoseKit training backend that takes a labeled PoseKit project + a pretrained ViTPose checkpoint and produces a validated fine-tuned checkpoint, launched as an in-env subprocess.

**Architecture:** Three layers along existing dependency boundaries. (1) A PoseKit-free **payload** subpackage of the Spec-1 leaf, `core/identity/pose/vitpose/training/`, holding the torch training loop, run as `python -m …vitpose.training --config run.json`. (2) Non-Qt **orchestration** in `posekit/core/` (checkpoint catalog + run-config assembly + subprocess command). (3) A thin Qt **worker** in the existing PoseKit training dialog.

**Tech Stack:** PyTorch (native, no mmcv/mmpose), NumPy, OpenCV, PySide/Qt (GUI worker only), pytest.

## Global Constraints

- **Leaf purity:** everything under `core/identity/pose/vitpose/` (now including `training/`) imports **nothing** from `hydra_suite`. Only relative sibling imports + torch/numpy/cv2. Enforced by an AST check (Task 9).
- **`core/` must not import `posekit/`.** Orchestration that calls `build_coco_keypoints_dataset` lives in `posekit/core/`, never in `core/`.
- **`vitpose/__init__.py` must NOT import `training/`** — a pure-inference `from …vitpose import build_vitpose` must never load the training loop.
- **Classic head only** (`build_head("classic", …)`). No MoE / ViTPose+ / `build_vitpose_moe`.
- **No flip augmentation.** Augmentation = scale + rotation + photometric only.
- **In-env subprocess:** launch with `sys.executable -m …`, `env=os.environ.copy()`. **No `conda run`** (that is the SLEAP-only path).
- **Variant keys are uppercase `S`/`B`/`L`/`H`** (`VARIANTS` dict), not lowercase.
- **Validation metric is PCK** (bbox-normalized), never OKS (no animal sigmas exist).
- **Checkpoints saved FP32.** `best.pt` selected on PCK@0.05.
- **Recipe defaults:** AdamW `lr=5e-4`, `wd=0.1`, grad-clip `1.0`, `sigma=2`, `drop_path=0.1` (override of the variant's inference default), `layer_decay` from `VARIANTS[variant].layer_decay`, input `256×192` (H×W), heatmap `64×48`, epochs default `40`.

### Verified leaf/PoseKit APIs (call these exactly)

From `hydra_suite.core.identity.pose.vitpose`:
- `config.py`: `IMAGE_SIZE_WH=(192,256)`, `HEATMAP_SIZE_WH=(48,64)`, `PIXEL_STD=200.0`, `PADDING_FACTOR=1.25`, `IMAGENET_MEAN`, `IMAGENET_STD`, `VARIANTS: dict[str, ViTPoseVariant]` (fields `embed_dim, depth, num_heads, part_features, drop_path_rate, layer_decay`).
- `model.py`: `ViT(embed_dim, depth, num_heads, drop_path_rate=0.0, part_features=…)` — **`build_vitpose` does NOT pass drop_path; assemble `ViT` directly to enable stochastic depth.**
- `heads.py`: `build_head(kind, embed_dim, num_keypoints)` → `ClassicHead` with `.deconv_layers` + `.final_layer = nn.Conv2d(256, K, 1)`.
- `vitpose.py`: `ViTPose(backbone, keypoint_head)`, `forward(x)->(N,K,64,48)`.
- `weights.py`: `CheckpointKeyError`; loader uses `torch.load(path, map_location="cpu", weights_only=True)`, unwraps `blob["state_dict"]` if present. `load_state_dict(strict=False)` **still raises on shape mismatch** → must pop `keypoint_head.final_layer.*` before load.
- `transforms.py`: `box2cs(box_xywh)->(center,scale)`; `affine_matrix(center,scale,rot)->2x3`; `top_down_affine(img,center,scale,rot)->warped (256x192)`; `normalize(img_bgr)->CHW float32`; `transform_preds(coords,center,scale,output_size_wh)->orig coords`.
- `decode.py`: `decode_udp_cv2(heatmaps, kernel=11)->(coords,maxvals)`; coords in **heatmap space** `(N,K,2)`; `get_max_preds`.

From `hydra_suite.posekit.core.extensions`:
- `build_coco_keypoints_dataset(image_paths, labels_dir, output_dir, class_names, keypoint_names, skeleton_edges, …)->{"dataset_dir","coco_path","labeled_count","manifest"}`. Writes `output_dir/images/` + `output_dir/annotations.json` (COCO keypoints: `images[{id,file_name,width,height}]`, `annotations[{id,image_id,category_id,bbox[x,y,w,h],area,iscrowd,num_keypoints,keypoints:[x,y,v]*K}]`, `categories[{id,name,keypoints[],skeleton[]}]`).

From `hydra_suite.posekit.gui.dialogs.utils`: `get_available_devices()->List[str]`.

Worker pattern to mirror — `TrainingWorker(QObject)` in `posekit/gui/dialogs/training.py`: signals `log=Signal(str)`, `progress=Signal(int,int)`, `finished=Signal(dict)`, `failed=Signal(str)`; `cancel()` sets a flag + `self._proc.terminate()`; `run()` does `subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1, env=os.environ.copy())` and streams stdout lines.

### File structure produced

```
core/identity/pose/vitpose/training/
  __init__.py        # minimal; NOT imported by vitpose/__init__.py
  config.py          # RunConfig dataclass + validation + json load/save
  targets.py         # UDP Gaussian heatmap encoder
  loss.py            # JointsMSELoss
  model_setup.py     # build_finetune_model, load_finetune_init, build_param_groups
  dataset.py         # CocoKeypointsDataset (+ augmentation, no flip)
  validate.py        # pck_batch, run_validation
  train.py           # train loop
  __main__.py        # CLI entry
posekit/core/vitpose_checkpoints.py    # fetch_pinned (HF), CATALOG, resolve_checkpoint
posekit/core/vitpose_training.py       # split, write/validate run.json, build command, parse progress
posekit/gui/dialogs/training.py        # + ViTPoseTrainingWorker, combo/controls wiring
tests/test_vitpose_finetune_*.py       # per-task tests
```

---

### Task 1: RunConfig schema (payload config)

**Files:**
- Create: `src/hydra_suite/core/identity/pose/vitpose/training/__init__.py`
- Create: `src/hydra_suite/core/identity/pose/vitpose/training/config.py`
- Test: `tests/test_vitpose_finetune_config.py`

**Interfaces:**
- Produces: `RunConfig` dataclass; `RunConfig.from_json(path: Path) -> RunConfig`; `RunConfig.to_json(path: Path) -> None`; `validate_run_config(d: dict) -> RunConfig` (raises `ValueError`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_finetune_config.py
import json
import pytest
from pathlib import Path
from hydra_suite.core.identity.pose.vitpose.training.config import (
    RunConfig, validate_run_config,
)

def _good(**over):
    d = dict(
        init_checkpoint="/tmp/x.pth", variant="B", num_keypoints=6,
        dataset_dir="/tmp/ds", output_dir="/tmp/run", device="cpu",
        epochs=40, batch_size=16, lr=5e-4, weight_decay=0.1,
        drop_path=0.1, sigma=2.0, grad_clip=1.0, val_fraction=0.2,
        seed=0, resume_from=None,
    )
    d.update(over)
    return d

def test_valid_config_roundtrips(tmp_path):
    cfg = validate_run_config(_good())
    assert cfg.variant == "B" and cfg.num_keypoints == 6
    p = tmp_path / "run.json"
    cfg.to_json(p)
    assert RunConfig.from_json(p).num_keypoints == 6

@pytest.mark.parametrize("over", [
    {"variant": "b"},            # lowercase rejected
    {"variant": "X"},            # unknown
    {"num_keypoints": 0},        # non-positive
    {"epochs": 0},               # non-positive
    {"val_fraction": 1.5},       # out of range
    {"unknown_key": 1},          # unknown key
])
def test_bad_config_rejected(over):
    with pytest.raises(ValueError):
        validate_run_config(_good(**over))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vitpose_finetune_config.py -q`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/core/identity/pose/vitpose/training/__init__.py
"""ViTPose fine-tuning payload. Imports nothing from hydra_suite (leaf-pure).
Run as: python -m hydra_suite.core.identity.pose.vitpose.training --config run.json
NOTE: never imported by vitpose/__init__.py."""
```

```python
# src/hydra_suite/core/identity/pose/vitpose/training/config.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import VARIANTS

_FIELDS = {
    "init_checkpoint", "variant", "num_keypoints", "dataset_dir", "output_dir",
    "device", "epochs", "batch_size", "lr", "weight_decay", "drop_path",
    "sigma", "grad_clip", "val_fraction", "seed", "resume_from",
}


@dataclass
class RunConfig:
    init_checkpoint: str
    variant: str
    num_keypoints: int
    dataset_dir: str
    output_dir: str
    device: str = "cpu"
    epochs: int = 40
    batch_size: int = 16
    lr: float = 5e-4
    weight_decay: float = 0.1
    drop_path: float = 0.1
    sigma: float = 2.0
    grad_clip: float = 1.0
    val_fraction: float = 0.2
    seed: int = 0
    resume_from: str | None = None

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "RunConfig":
        return validate_run_config(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_run_config(d: dict) -> RunConfig:
    unknown = set(d) - _FIELDS
    if unknown:
        raise ValueError(f"unknown run.json keys: {sorted(unknown)}")
    if d.get("variant") not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)} (uppercase); got {d.get('variant')!r}")
    if int(d.get("num_keypoints", 0)) <= 0:
        raise ValueError("num_keypoints must be positive")
    if int(d.get("epochs", 0)) <= 0:
        raise ValueError("epochs must be positive")
    vf = float(d.get("val_fraction", 0.2))
    if not (0.0 < vf < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")
    return RunConfig(**d)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vitpose_finetune_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/identity/pose/vitpose/training/__init__.py \
        src/hydra_suite/core/identity/pose/vitpose/training/config.py \
        tests/test_vitpose_finetune_config.py
git commit -m "feat(vitpose-finetune): run.json config schema + validation"
```

---

### Task 2: UDP Gaussian target encoder

The one genuinely new numerical component (Spec 1 built *decode*; this is the *encode* inverse). Gate: encode→decode ≈ identity within the Spec-1 UDP float floor.

**Files:**
- Create: `src/hydra_suite/core/identity/pose/vitpose/training/targets.py`
- Test: `tests/test_vitpose_finetune_targets.py`

**Interfaces:**
- Produces: `generate_udp_gaussian(joints_hm: np.ndarray, vis: np.ndarray, heatmap_size_wh: tuple[int,int], sigma: float) -> tuple[np.ndarray, np.ndarray]`. `joints_hm` is `(K,2)` in **heatmap** coords; returns `target (K,H,W) float32` and `target_weight (K,1) float32`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_finetune_targets.py
import numpy as np
from hydra_suite.core.identity.pose.vitpose.training.targets import generate_udp_gaussian
from hydra_suite.core.identity.pose.vitpose.decode import decode_udp_cv2

HM_WH = (48, 64)  # (W, H)

def test_encode_decode_roundtrip_subpixel():
    # subpixel centers, comfortably inside the map
    joints = np.array([[10.3, 20.7], [30.9, 40.1], [5.0, 5.0]], dtype=np.float32)
    vis = np.ones(3, dtype=np.float32)
    target, weight = generate_udp_gaussian(joints, vis, HM_WH, sigma=2.0)
    assert target.shape == (3, 64, 48)
    assert weight.shape == (3, 1)
    coords, maxvals = decode_udp_cv2(target[None, ...], kernel=11)  # (1,K,2)
    rec = coords[0]
    assert np.allclose(rec, joints, atol=0.25), f"{rec} vs {joints}"

def test_invisible_joint_zeroed():
    joints = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    vis = np.array([1.0, 0.0], dtype=np.float32)
    target, weight = generate_udp_gaussian(joints, vis, HM_WH, sigma=2.0)
    assert weight[1, 0] == 0.0
    assert target[1].max() == 0.0
    assert target[0].max() > 0.9  # peak ~1.0 at the visible joint
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vitpose_finetune_targets.py -q`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/core/identity/pose/vitpose/training/targets.py
from __future__ import annotations

import numpy as np


def generate_udp_gaussian(
    joints_hm: np.ndarray,
    vis: np.ndarray,
    heatmap_size_wh: tuple[int, int],
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """UDP GaussianHeatmap target: full-map Gaussian at the subpixel joint
    center (mmpose encoding='UDP', target_type='GaussianHeatmap').

    joints_hm: (K, 2) keypoint coords already in heatmap pixel space.
    vis: (K,) visibility (>0 => labelled).
    Returns target (K, H, W) float32 and target_weight (K, 1) float32.
    """
    w, h = heatmap_size_wh
    k = joints_hm.shape[0]
    target = np.zeros((k, h, w), dtype=np.float32)
    weight = (np.asarray(vis).reshape(k) > 0).astype(np.float32).reshape(k, 1)
    xs = np.arange(w, dtype=np.float32)[None, :]      # (1, W)
    ys = np.arange(h, dtype=np.float32)[:, None]      # (H, 1)
    two_s2 = 2.0 * sigma * sigma
    for j in range(k):
        if weight[j, 0] == 0.0:
            continue
        mu_x, mu_y = float(joints_hm[j, 0]), float(joints_hm[j, 1])
        target[j] = np.exp(-(((xs - mu_x) ** 2) + ((ys - mu_y) ** 2)) / two_s2)
    return target, weight
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vitpose_finetune_targets.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/identity/pose/vitpose/training/targets.py tests/test_vitpose_finetune_targets.py
git commit -m "feat(vitpose-finetune): UDP Gaussian heatmap target encoder (round-trip verified)"
```

---

### Task 3: JointsMSELoss

**Files:**
- Create: `src/hydra_suite/core/identity/pose/vitpose/training/loss.py`
- Test: `tests/test_vitpose_finetune_loss.py`

**Interfaces:**
- Produces: `JointsMSELoss(use_target_weight: bool = True)`, `forward(output, target, target_weight) -> scalar tensor`. `output/target` are `(B,K,H,W)`; `target_weight` is `(B,K,1)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_finetune_loss.py
import torch
from hydra_suite.core.identity.pose.vitpose.training.loss import JointsMSELoss

def test_zero_loss_when_equal():
    out = torch.rand(2, 3, 8, 6)
    w = torch.ones(2, 3, 1)
    loss = JointsMSELoss(True)(out, out.clone(), w)
    assert loss.item() < 1e-8

def test_weight_masks_joint():
    out = torch.zeros(1, 2, 4, 4)
    tgt = torch.zeros(1, 2, 4, 4)
    out[0, 1] = 5.0  # only joint 1 wrong
    w = torch.tensor([[[1.0], [0.0]]])  # joint 1 masked out
    assert JointsMSELoss(True)(out, tgt, w).item() < 1e-8
    w2 = torch.ones(1, 2, 1)
    assert JointsMSELoss(True)(out, tgt, w2).item() > 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vitpose_finetune_loss.py -q`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/core/identity/pose/vitpose/training/loss.py
from __future__ import annotations

import torch
import torch.nn as nn


class JointsMSELoss(nn.Module):
    """Per-joint heatmap MSE with optional per-joint visibility weighting.
    Mirrors mmpose JointsMSELoss(use_target_weight=True)."""

    def __init__(self, use_target_weight: bool = True) -> None:
        super().__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        self.use_target_weight = use_target_weight

    def forward(
        self, output: torch.Tensor, target: torch.Tensor, target_weight: torch.Tensor
    ) -> torch.Tensor:
        b, k = output.shape[0], output.shape[1]
        pred = output.reshape(b, k, -1)
        gt = target.reshape(b, k, -1)
        loss = output.new_zeros(())
        for j in range(k):
            pj, gj = pred[:, j], gt[:, j]
            if self.use_target_weight:
                w = target_weight[:, j]  # (B,1)
                loss = loss + 0.5 * self.criterion(pj * w, gj * w)
            else:
                loss = loss + 0.5 * self.criterion(pj, gj)
        return loss / k
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vitpose_finetune_loss.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/identity/pose/vitpose/training/loss.py tests/test_vitpose_finetune_loss.py
git commit -m "feat(vitpose-finetune): JointsMSELoss with per-joint weighting"
```

---

### Task 4: Model assembly + fine-tune checkpoint loader

**Files:**
- Create: `src/hydra_suite/core/identity/pose/vitpose/training/model_setup.py`
- Test: `tests/test_vitpose_finetune_model_setup.py`

**Interfaces:**
- Produces: `build_finetune_model(variant: str, num_keypoints: int, drop_path: float) -> ViTPose`; `load_finetune_init(model: ViTPose, ckpt_path: Path) -> None` (backbone strict, head re-init; raises `CheckpointKeyError` unless the only missing keys are `keypoint_head.final_layer.{weight,bias}`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_finetune_model_setup.py
import torch
import pytest
from hydra_suite.core.identity.pose.vitpose.training.model_setup import (
    build_finetune_model, load_finetune_init,
)
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose
from hydra_suite.core.identity.pose.vitpose.weights import CheckpointKeyError

def test_build_shapes_and_droppath():
    m = build_finetune_model("B", num_keypoints=6, drop_path=0.1)
    out = m(torch.zeros(1, 3, 256, 192))
    assert out.shape == (1, 6, 64, 48)

def test_load_reinits_only_final_layer(tmp_path):
    # a "pretrained" classic K=17 checkpoint, saved as {"state_dict": ...}
    pre = build_vitpose("B", "classic", num_keypoints=17)
    ckpt = tmp_path / "pre.pth"
    torch.save({"state_dict": pre.state_dict()}, ckpt)

    model = build_finetune_model("B", num_keypoints=6, drop_path=0.1)
    fresh_final = model.keypoint_head.final_layer.weight.clone()
    # a backbone param must actually change to the pretrained value
    load_finetune_init(model, ckpt)
    assert torch.equal(
        model.backbone.blocks[0].attn.qkv.weight, pre.backbone.blocks[0].attn.qkv.weight
    )
    # final layer stays the freshly-initialised K=6 conv (shape 6, not 17)
    assert model.keypoint_head.final_layer.weight.shape[0] == 6
    assert torch.equal(model.keypoint_head.final_layer.weight, fresh_final)

def test_load_rejects_variant_mismatch(tmp_path):
    pre = build_vitpose("S", "classic", num_keypoints=17)  # wrong variant
    ckpt = tmp_path / "s.pth"
    torch.save(pre.state_dict(), ckpt)
    model = build_finetune_model("B", num_keypoints=6, drop_path=0.1)
    with pytest.raises(CheckpointKeyError):
        load_finetune_init(model, ckpt)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vitpose_finetune_model_setup.py -q`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/core/identity/pose/vitpose/training/model_setup.py
from __future__ import annotations

from pathlib import Path

import torch

from ..config import VARIANTS
from ..heads import build_head
from ..model import ViT
from ..vitpose import ViTPose
from ..weights import CheckpointKeyError

_EXPECTED_MISSING = {"keypoint_head.final_layer.weight", "keypoint_head.final_layer.bias"}


def build_finetune_model(variant: str, num_keypoints: int, drop_path: float) -> ViTPose:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r} (expected one of {sorted(VARIANTS)})")
    v = VARIANTS[variant]
    backbone = ViT(
        embed_dim=v.embed_dim, depth=v.depth, num_heads=v.num_heads, drop_path_rate=drop_path
    )
    head = build_head("classic", v.embed_dim, num_keypoints)
    return ViTPose(backbone, head)


def load_finetune_init(model: ViTPose, ckpt_path: Path) -> None:
    """Load a pretrained ViTPose checkpoint for fine-tuning: backbone (and head
    deconv) load strict; `keypoint_head.final_layer` is left freshly initialised
    so K can differ. Raises unless the ONLY missing keys are final_layer.*."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    cleaned = {
        key: val
        for key, val in state.items()
        if not key.startswith("keypoint_head.final_layer.")
        and not key.startswith("associate_keypoint_heads.")
    }
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    missing, unexpected = set(missing), set(unexpected)
    if missing != _EXPECTED_MISSING or unexpected:
        raise CheckpointKeyError(
            f"fine-tune load mismatch for {Path(ckpt_path).name}\n"
            f"  unexpected missing: {sorted(missing - _EXPECTED_MISSING)}\n"
            f"  not-actually-missing: {sorted(_EXPECTED_MISSING - missing)}\n"
            f"  unexpected keys: {sorted(unexpected)[:10]}"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vitpose_finetune_model_setup.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/identity/pose/vitpose/training/model_setup.py tests/test_vitpose_finetune_model_setup.py
git commit -m "feat(vitpose-finetune): model assembly with drop_path + backbone-strict fine-tune loader"
```

---

### Task 5: Layer-wise LR-decay param groups

The `+2` is the roadmap's named trap: `num_layers = depth + 2`; `lr_scale = layer_decay ** (num_layers - layer_id - 1)`.

**Files:**
- Modify: `src/hydra_suite/core/identity/pose/vitpose/training/model_setup.py` (add `build_param_groups`)
- Test: `tests/test_vitpose_finetune_param_groups.py`

**Interfaces:**
- Produces: `build_param_groups(model: ViTPose, base_lr: float, layer_decay: float, weight_decay: float) -> list[dict]` (each dict has `params`, `lr`, `weight_decay`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_finetune_param_groups.py
import math
import torch
from hydra_suite.core.identity.pose.vitpose.training.model_setup import (
    build_finetune_model, build_param_groups, _layer_id_for,
)

def test_layer_ids_use_plus_two():
    m = build_finetune_model("B", 6, 0.1)  # depth=12 -> num_layers=14
    num_layers = 12 + 2
    assert _layer_id_for("backbone.pos_embed", num_layers) == 0
    assert _layer_id_for("backbone.patch_embed.proj.weight", num_layers) == 0
    assert _layer_id_for("backbone.blocks.0.attn.qkv.weight", num_layers) == 1
    assert _layer_id_for("backbone.blocks.11.mlp.fc1.weight", num_layers) == 12
    assert _layer_id_for("keypoint_head.final_layer.weight", num_layers) == num_layers - 1
    assert _layer_id_for("backbone.last_norm.weight", num_layers) == num_layers - 1

def test_lr_scales_and_no_decay():
    m = build_finetune_model("B", 6, 0.1)
    decay, base_lr, wd = 0.75, 5e-4, 0.1
    groups = build_param_groups(m, base_lr, decay, wd)
    # every parameter appears exactly once
    n_params = sum(len(g["params"]) for g in groups)
    assert n_params == sum(1 for _ in m.parameters())
    # head group runs at full base_lr (scale ** 0)
    head_lrs = [g["lr"] for g in groups if any(
        p is m.keypoint_head.final_layer.weight for p in g["params"])]
    assert math.isclose(head_lrs[0], base_lr, rel_tol=1e-6)
    # bias / norm / pos_embed groups carry zero weight decay
    for g in groups:
        if g["weight_decay"] == 0.0:
            continue
        for p in g["params"]:
            assert p.ndim > 1  # decayed params are weight matrices, never biases
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vitpose_finetune_param_groups.py -q`
Expected: FAIL (`build_param_groups`/`_layer_id_for` undefined).

- [ ] **Step 3: Write minimal implementation** (append to `model_setup.py`)

```python
def _layer_id_for(name: str, num_layers: int) -> int:
    if name.startswith("backbone.patch_embed") or name == "backbone.pos_embed":
        return 0
    if name.startswith("backbone.blocks."):
        return int(name.split(".")[2]) + 1
    return num_layers - 1  # head, last_norm, everything downstream


def _no_decay(name: str, param) -> bool:
    return param.ndim <= 1 or name.endswith("pos_embed")


def build_param_groups(model, base_lr: float, layer_decay: float, weight_decay: float) -> list[dict]:
    depth = len(model.backbone.blocks)
    num_layers = depth + 2
    buckets: dict[tuple[int, bool], dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lid = _layer_id_for(name, num_layers)
        scale = layer_decay ** (num_layers - lid - 1)
        decayed = not _no_decay(name, param)
        key = (lid, decayed)
        if key not in buckets:
            buckets[key] = {
                "params": [],
                "lr": base_lr * scale,
                "weight_decay": weight_decay if decayed else 0.0,
            }
        buckets[key]["params"].append(param)
    return list(buckets.values())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vitpose_finetune_param_groups.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/identity/pose/vitpose/training/model_setup.py tests/test_vitpose_finetune_param_groups.py
git commit -m "feat(vitpose-finetune): layer-wise LR-decay param groups (num_layers=depth+2)"
```

---

### Task 6: COCO-keypoints dataset (+ augmentation)

**Files:**
- Create: `src/hydra_suite/core/identity/pose/vitpose/training/dataset.py`
- Test: `tests/test_vitpose_finetune_dataset.py`

**Interfaces:**
- Consumes: `generate_udp_gaussian` (Task 2); leaf `box2cs`, `affine_matrix`, `top_down_affine`, `normalize`; `IMAGE_SIZE_WH`, `HEATMAP_SIZE_WH`.
- Produces: `CocoKeypointsDataset(dataset_dir: Path, ids: list[int], sigma: float, augment: bool)`; `__getitem__` returns dict `{"image": (3,256,192) f32 tensor, "target": (K,64,48) f32 tensor, "target_weight": (K,1) f32 tensor, "center": (2,), "scale": (2,), "gt_joints": (K,3) orig-space xyv, "bbox": (4,), "image_id": int}`. Module-level `FEAT_STRIDE: np.ndarray`. Helper `load_coco_index(dataset_dir) -> tuple[list[int], dict]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_finetune_dataset.py
import json
import numpy as np
import cv2
import torch
from pathlib import Path
from hydra_suite.core.identity.pose.vitpose.training.dataset import (
    CocoKeypointsDataset, load_coco_index, FEAT_STRIDE,
)

def _make_ds(tmp_path, k=3):
    (tmp_path / "images").mkdir()
    img = np.full((100, 80, 3), 127, np.uint8)
    cv2.imwrite(str(tmp_path / "images" / "f0.png"), img)
    kpts = []
    for j in range(k):
        kpts += [20 + 5 * j, 30 + 5 * j, 2]
    coco = {
        "images": [{"id": 1, "file_name": "f0.png", "width": 80, "height": 100}],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 1,
            "bbox": [10.0, 10.0, 40.0, 60.0], "area": 2400.0, "iscrowd": 0,
            "num_keypoints": k, "keypoints": kpts,
        }],
        "categories": [{"id": 1, "name": "a", "keypoints": [f"k{j}" for j in range(k)], "skeleton": []}],
    }
    (tmp_path / "annotations.json").write_text(json.dumps(coco))
    return tmp_path

def test_getitem_shapes(tmp_path):
    ds_dir = _make_ds(tmp_path)
    ids, _ = load_coco_index(ds_dir)
    ds = CocoKeypointsDataset(ds_dir, ids, sigma=2.0, augment=False)
    s = ds[0]
    assert s["image"].shape == (3, 256, 192)
    assert s["target"].shape == (3, 64, 48)
    assert s["target_weight"].shape == (3, 1)
    assert torch.all(s["target_weight"] == 1.0)

def test_target_peak_matches_decoded_gt(tmp_path):
    # With no augmentation, decoding the GT heatmap and mapping back through
    # transform_preds must recover the annotated keypoints (sub-pixel).
    from hydra_suite.core.identity.pose.vitpose.decode import decode_udp_cv2
    from hydra_suite.core.identity.pose.vitpose.transforms import transform_preds
    ds_dir = _make_ds(tmp_path)
    ids, _ = load_coco_index(ds_dir)
    ds = CocoKeypointsDataset(ds_dir, ids, sigma=2.0, augment=False)
    s = ds[0]
    coords, _ = decode_udp_cv2(s["target"].numpy()[None], kernel=11)
    orig = transform_preds(coords[0], s["center"].numpy(), s["scale"].numpy(), (48, 64))
    gt = s["gt_joints"].numpy()[:, :2]
    assert np.allclose(orig, gt, atol=1.0)

def test_feat_stride_value():
    assert np.allclose(FEAT_STRIDE, (np.array([192, 256]) - 1.0) / (np.array([48, 64]) - 1.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vitpose_finetune_dataset.py -q`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/core/identity/pose/vitpose/training/dataset.py
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import HEATMAP_SIZE_WH, IMAGE_SIZE_WH
from ..transforms import affine_matrix, box2cs, normalize, top_down_affine
from .targets import generate_udp_gaussian

FEAT_STRIDE = (np.array(IMAGE_SIZE_WH, np.float32) - 1.0) / (
    np.array(HEATMAP_SIZE_WH, np.float32) - 1.0
)

# augmentation ranges (no flip, per spec)
_SCALE_JITTER = 0.30
_ROT_RANGE = 40.0
_ROT_PROB = 0.6


def load_coco_index(dataset_dir: Path) -> tuple[list[int], dict]:
    coco = json.loads((Path(dataset_dir) / "annotations.json").read_text(encoding="utf-8"))
    images = {img["id"]: img for img in coco["images"]}
    anns = [a for a in coco["annotations"] if a.get("num_keypoints", 0) > 0]
    index = {a["id"]: (a, images[a["image_id"]]) for a in anns}
    return list(index.keys()), index


def _warp_joints(joints_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return joints_xy @ matrix[:, :2].T + matrix[:, 2]


class CocoKeypointsDataset(Dataset):
    def __init__(self, dataset_dir: Path, ids: list[int], sigma: float, augment: bool) -> None:
        self.dir = Path(dataset_dir)
        self.ids, self.index = load_coco_index(dataset_dir)
        self.ids = [i for i in ids if i in self.index]
        self.sigma = float(sigma)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict:
        ann, img_meta = self.index[self.ids[i]]
        img = cv2.imread(str(self.dir / "images" / img_meta["file_name"]), cv2.IMREAD_COLOR)
        kp = np.array(ann["keypoints"], np.float32).reshape(-1, 3)
        k = kp.shape[0]
        center, scale = box2cs(np.array(ann["bbox"], np.float32))

        rot = 0.0
        if self.augment:
            scale = scale * float(np.clip(np.random.randn() * 0.25 + 1.0, 1 - _SCALE_JITTER, 1 + _SCALE_JITTER))
            if np.random.rand() < _ROT_PROB:
                rot = float(np.clip(np.random.randn() * (_ROT_RANGE / 2), -_ROT_RANGE, _ROT_RANGE))

        warped = top_down_affine(img, center, scale, rot)
        if self.augment:
            warped = _photometric(warped)
        image = torch.from_numpy(normalize(warped))

        matrix = affine_matrix(center, scale, rot)
        joints_in = _warp_joints(kp[:, :2], matrix)          # input-crop space
        joints_hm = joints_in / FEAT_STRIDE                   # heatmap space
        vis = kp[:, 2]
        target, weight = generate_udp_gaussian(joints_hm, vis, HEATMAP_SIZE_WH, self.sigma)

        return {
            "image": image,
            "target": torch.from_numpy(target),
            "target_weight": torch.from_numpy(weight),
            "center": torch.from_numpy(center),
            "scale": torch.from_numpy(scale),
            "gt_joints": torch.from_numpy(kp),
            "bbox": torch.tensor(ann["bbox"], dtype=torch.float32),
            "image_id": int(ann["image_id"]),
        }


def _photometric(img_bgr: np.ndarray) -> np.ndarray:
    out = img_bgr.astype(np.float32)
    out *= np.random.uniform(0.7, 1.3)               # brightness
    mean = out.mean(axis=(0, 1), keepdims=True)
    out = (out - mean) * np.random.uniform(0.7, 1.3) + mean  # contrast
    return np.clip(out, 0, 255).astype(np.uint8)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vitpose_finetune_dataset.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/identity/pose/vitpose/training/dataset.py tests/test_vitpose_finetune_dataset.py
git commit -m "feat(vitpose-finetune): COCO-keypoints dataset with UDP targets + no-flip augmentation"
```

---

### Task 7: PCK metric + validation

**Files:**
- Create: `src/hydra_suite/core/identity/pose/vitpose/training/validate.py`
- Test: `tests/test_vitpose_finetune_validate.py`

**Interfaces:**
- Consumes: leaf `decode_udp_cv2`, `transform_preds`; `HEATMAP_SIZE_WH`.
- Produces: `pck_from_preds(pred_xy, gt_xyv, bbox, thresholds) -> dict[float, float]`; `run_validation(model, loader, device, thresholds=(0.05, 0.1)) -> dict` returning `{"val_loss": float, "pck": {0.05: .., 0.1: ..}}`. Normalization = `sqrt(bbox_w * bbox_h)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_finetune_validate.py
import numpy as np
from hydra_suite.core.identity.pose.vitpose.training.validate import pck_from_preds

def test_pck_perfect_and_thresholded():
    gt = np.array([[10, 10, 2], [20, 20, 2]], np.float32)
    bbox = np.array([0, 0, 40, 40], np.float32)  # norm = sqrt(1600) = 40
    perfect = gt[:, :2].copy()
    assert pck_from_preds(perfect, gt, bbox, (0.05, 0.1))[0.05] == 1.0
    # move one joint 3 px: 3/40 = 0.075 -> fails @0.05, passes @0.1
    off = gt[:, :2].copy(); off[0, 0] += 3.0
    r = pck_from_preds(off, gt, bbox, (0.05, 0.1))
    assert r[0.05] == 0.5 and r[0.1] == 1.0

def test_pck_ignores_invisible():
    gt = np.array([[10, 10, 2], [20, 20, 0]], np.float32)  # joint 1 unlabelled
    bbox = np.array([0, 0, 40, 40], np.float32)
    pred = np.array([[10, 10], [999, 999]], np.float32)
    assert pck_from_preds(pred, gt, bbox, (0.05,))[0.05] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vitpose_finetune_validate.py -q`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/core/identity/pose/vitpose/training/validate.py
from __future__ import annotations

import numpy as np
import torch

from ..config import HEATMAP_SIZE_WH
from ..decode import decode_udp_cv2
from ..transforms import transform_preds
from .loss import JointsMSELoss


def pck_from_preds(pred_xy, gt_xyv, bbox, thresholds) -> dict:
    gt_xyv = np.asarray(gt_xyv, np.float32)
    vis = gt_xyv[:, 2] > 0
    norm = float(np.sqrt(max(bbox[2] * bbox[3], 1e-6)))
    dist = np.linalg.norm(pred_xy[vis] - gt_xyv[vis, :2], axis=1) / norm
    out = {}
    for t in thresholds:
        out[t] = float((dist < t).mean()) if vis.any() else 0.0
    return out


def run_validation(model, loader, device, thresholds=(0.05, 0.1)) -> dict:
    model.eval()
    crit = JointsMSELoss(True)
    total_loss, n = 0.0, 0
    acc = {t: [] for t in thresholds}
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            out = model(img)
            total_loss += crit(
                out.cpu(), batch["target"], batch["target_weight"]
            ).item() * img.shape[0]
            n += img.shape[0]
            hm = out.cpu().numpy()
            coords, _ = decode_udp_cv2(hm, kernel=11)  # (B,K,2) heatmap space
            for b in range(img.shape[0]):
                pred = transform_preds(
                    coords[b], batch["center"][b].numpy(), batch["scale"][b].numpy(), HEATMAP_SIZE_WH
                )
                r = pck_from_preds(
                    pred, batch["gt_joints"][b].numpy(), batch["bbox"][b].numpy(), thresholds
                )
                for t in thresholds:
                    acc[t].append(r[t])
    return {
        "val_loss": total_loss / max(n, 1),
        "pck": {t: float(np.mean(v)) if v else 0.0 for t, v in acc.items()},
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vitpose_finetune_validate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/identity/pose/vitpose/training/validate.py tests/test_vitpose_finetune_validate.py
git commit -m "feat(vitpose-finetune): bbox-normalized PCK metric + validation loop"
```

---

### Task 8: Training loop + CLI (the objective overfit gate)

**Files:**
- Create: `src/hydra_suite/core/identity/pose/vitpose/training/train.py`
- Create: `src/hydra_suite/core/identity/pose/vitpose/training/__main__.py`
- Test: `tests/test_vitpose_finetune_train.py`

**Interfaces:**
- Consumes: all prior payload modules; `RunConfig` (Task 1).
- Produces: `train(cfg: RunConfig) -> dict` (returns `{"best_pck": float, "best_epoch": int, "output_dir": str}`); writes `output_dir/{last.pt,best.pt,metrics.csv,run.json}` + `output_dir/val_overlays/*.png`. Emits one stdout line per epoch: `EPOCH {e} train_loss={:.5f} val_loss={:.5f} pck@0.05={:.4f} pck@0.1={:.4f}`. `best.pt` payload: `{"model_state","variant","num_keypoints","epoch","pck"}` where `model_state` matches `build_vitpose(variant,"classic",K)`.

- [ ] **Step 1: Write the failing test** (tiny overfit — Spec 4's objective gate)

```python
# tests/test_vitpose_finetune_train.py
import json
import numpy as np
import cv2
import torch
from pathlib import Path
from hydra_suite.core.identity.pose.vitpose.training.config import RunConfig
from hydra_suite.core.identity.pose.vitpose.training.train import train

def _tiny_dataset(root: Path, n=8, k=3):
    (root / "images").mkdir(parents=True)
    images, anns = [], []
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.integers(0, 255, (100, 80, 3), dtype=np.uint8)
        cv2.imwrite(str(root / "images" / f"f{i}.png"), img)
        kp = []
        for j in range(k):
            kp += [25 + 4 * j, 30 + 4 * j, 2]
        images.append({"id": i + 1, "file_name": f"f{i}.png", "width": 80, "height": 100})
        anns.append({
            "id": i + 1, "image_id": i + 1, "category_id": 1,
            "bbox": [10.0, 10.0, 50.0, 70.0], "area": 3500.0, "iscrowd": 0,
            "num_keypoints": k, "keypoints": kp,
        })
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "a", "keypoints": [f"k{j}" for j in range(k)], "skeleton": []}]}
    (root / "annotations.json").write_text(json.dumps(coco))

def test_tiny_overfit_drives_metrics(tmp_path):
    # Variant "S" (not "B") keeps this CPU gate fast; the loop/targets/loss are
    # architecture-agnostic, so S exercises exactly the same code paths.
    ds = tmp_path / "ds"; _tiny_dataset(ds)
    # a random-init "pretrained" checkpoint so the loader path is exercised
    from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose
    pre = tmp_path / "pre.pth"
    torch.save({"state_dict": build_vitpose("S", "classic", 17).state_dict()}, pre)

    out = tmp_path / "run"
    cfg = RunConfig(
        init_checkpoint=str(pre), variant="S", num_keypoints=3,
        dataset_dir=str(ds), output_dir=str(out), device="cpu",
        epochs=3, batch_size=4, lr=1e-3, val_fraction=0.25, drop_path=0.0, seed=0,
    )
    result = train(cfg)
    assert (out / "best.pt").exists()
    assert (out / "metrics.csv").exists()
    # loss must have decreased across the 3 epochs
    rows = (out / "metrics.csv").read_text().strip().splitlines()[1:]
    losses = [float(r.split(",")[1]) for r in rows]
    assert losses[-1] < losses[0]
    # best.pt loads back into a K=3 classic model
    blob = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    assert blob["num_keypoints"] == 3
    build_vitpose("S", "classic", 3).load_state_dict(blob["model_state"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vitpose_finetune_train.py -q`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/core/identity/pose/vitpose/training/train.py
from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import HEATMAP_SIZE_WH, VARIANTS
from ..decode import decode_udp_cv2
from ..transforms import transform_preds
from .config import RunConfig
from .dataset import CocoKeypointsDataset, load_coco_index
from .loss import JointsMSELoss
from .model_setup import build_finetune_model, build_param_groups, load_finetune_init
from .validate import run_validation


def _split(ids: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    order = ids[:]
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * val_fraction)))
    return order[n_val:], order[:n_val]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(cfg: RunConfig) -> dict:
    _seed_everything(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out_dir / "run.json")
    device = torch.device(cfg.device)

    all_ids, _ = load_coco_index(Path(cfg.dataset_dir))
    train_ids, val_ids = _split(all_ids, cfg.val_fraction, cfg.seed)
    train_ds = CocoKeypointsDataset(cfg.dataset_dir, train_ids, cfg.sigma, augment=True)
    val_ds = CocoKeypointsDataset(cfg.dataset_dir, val_ids, cfg.sigma, augment=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = build_finetune_model(cfg.variant, cfg.num_keypoints, cfg.drop_path)
    load_finetune_init(model, Path(cfg.init_checkpoint))
    model.to(device)

    groups = build_param_groups(model, cfg.lr, VARIANTS[cfg.variant].layer_decay, cfg.weight_decay)
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    crit = JointsMSELoss(True)

    start_epoch, best_pck, best_epoch = 0, -1.0, 0
    if cfg.resume_from:
        ckpt = torch.load(cfg.resume_from, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optim_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_pck = float(ckpt.get("pck", -1.0))

    metrics_path = out_dir / "metrics.csv"
    if start_epoch == 0:
        metrics_path.write_text("epoch,train_loss,val_loss,pck@0.05,pck@0.1\n", encoding="utf-8")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
            img = batch["image"].to(device)
            out = model(img)
            loss = crit(out, batch["target"].to(device), batch["target_weight"].to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += loss.item() * img.shape[0]
            n += img.shape[0]
        sched.step()
        train_loss = running / max(n, 1)

        val = run_validation(model, val_loader, device)
        p05, p10 = val["pck"][0.05], val["pck"][0.1]
        print(f"EPOCH {epoch} train_loss={train_loss:.5f} val_loss={val['val_loss']:.5f} "
              f"pck@0.05={p05:.4f} pck@0.1={p10:.4f}", flush=True)
        with metrics_path.open("a", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerow([epoch, f"{train_loss:.6f}", f"{val['val_loss']:.6f}",
                                     f"{p05:.6f}", f"{p10:.6f}"])

        ckpt = {"model_state": model.state_dict(), "optim_state": opt.state_dict(),
                "variant": cfg.variant, "num_keypoints": cfg.num_keypoints,
                "epoch": epoch, "pck": p05}
        torch.save(ckpt, out_dir / "last.pt")
        if p05 >= best_pck:
            best_pck, best_epoch = p05, epoch
            torch.save(ckpt, out_dir / "best.pt")

    _write_val_overlays(model, val_ds, device, out_dir / "val_overlays", cfg.num_keypoints)
    return {"best_pck": best_pck, "best_epoch": best_epoch, "output_dir": str(out_dir)}


def _write_val_overlays(model, val_ds, device, dst: Path, k: int, limit: int = 6) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i in range(min(limit, len(val_ds))):
            s = val_ds[i]
            out = model(s["image"].unsqueeze(0).to(device)).cpu().numpy()
            coords, _ = decode_udp_cv2(out, kernel=11)
            pred = transform_preds(coords[0], s["center"].numpy(), s["scale"].numpy(), HEATMAP_SIZE_WH)
            img = cv2.imread(str(val_ds.dir / "images" /
                                 val_ds.index[val_ds.ids[i]][1]["file_name"]))
            for j in range(k):
                cv2.circle(img, (int(pred[j, 0]), int(pred[j, 1])), 3, (0, 0, 255), -1)
            cv2.imwrite(str(dst / f"val_{i}.png"), img)
```

```python
# src/hydra_suite/core/identity/pose/vitpose/training/__main__.py
from __future__ import annotations

import argparse
from pathlib import Path

from .config import RunConfig
from .train import train


def main() -> None:
    ap = argparse.ArgumentParser(prog="vitpose.training")
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    result = train(RunConfig.from_json(args.config))
    print(f"DONE best_pck={result['best_pck']:.4f} best_epoch={result['best_epoch']}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vitpose_finetune_train.py -q`
Expected: PASS (train loss decreases; `best.pt` reloads into a K=3 model).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/identity/pose/vitpose/training/train.py \
        src/hydra_suite/core/identity/pose/vitpose/training/__main__.py \
        tests/test_vitpose_finetune_train.py
git commit -m "feat(vitpose-finetune): training loop, CLI, resume, val overlays (tiny-overfit gate)"
```

---

### Task 9: Extend leaf-purity check to cover `training/`

**Files:**
- Modify: the existing leaf-purity test (find it: `grep -rl "imports nothing from hydra_suite\|leaf" tests/`; it AST-scans `vitpose/*.py`).
- Test: same file (extend it).

**Interfaces:**
- Consumes: nothing new. Ensures every `.py` under `vitpose/` (recursively, including `training/`) has zero `import hydra_suite` / `from hydra_suite …` nodes; and asserts `vitpose/__init__.py` does not import the `training` subpackage.

- [ ] **Step 1: Write the failing test** — locate the existing check and add:

```python
# in the existing leaf-purity test module
import ast
from pathlib import Path

VITPOSE = Path("src/hydra_suite/core/identity/pose/vitpose")

def test_training_subpackage_is_leaf_pure():
    offenders = []
    for py in VITPOSE.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name.split(".")[0] == "hydra_suite" for a in node.names):
                    offenders.append(str(py))
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] == "hydra_suite" and node.level == 0:
                    offenders.append(str(py))
    assert not offenders, f"leaf-impure files: {sorted(set(offenders))}"

def test_init_does_not_eager_import_training():
    src = (VITPOSE / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("training"):
            raise AssertionError("vitpose/__init__.py must not import training/")
        if isinstance(node, ast.ImportFrom) and node.level > 0 and (node.module or "") == "training":
            raise AssertionError("vitpose/__init__.py must not import training/")
```

- [ ] **Step 2: Run test to verify it passes now** (the payload is already leaf-pure by construction)

Run: `pytest tests/ -k "leaf_pure or eager_import" -q`
Expected: PASS. If it FAILS, a payload file imported `hydra_suite` — fix the payload, not the test.

- [ ] **Step 3: (guard) Prove the check bites** — temporarily add `import hydra_suite` to `training/loss.py`, re-run, confirm FAIL, then revert.

- [ ] **Step 4: Re-run to confirm PASS after revert.**

Run: `pytest tests/ -k "leaf_pure or eager_import" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test(vitpose-finetune): extend leaf-purity AST check to training/ subpackage"
```

---

### Task 10: Base-checkpoint catalog + SHA-pinned downloader (orchestration)

**Files:**
- Create: `src/hydra_suite/posekit/core/vitpose_checkpoints.py`
- Test: `tests/test_vitpose_checkpoints.py`

**Mechanism note:** the ViTPose weights are re-hosted on HuggingFace
(`nielsr/vitpose-original-checkpoints`) and fetched via `hf_hub_download` — this
is exactly what `tools/vitpose/fetch_assets.py` already does. Use that mechanism
(NOT raw-URL `urlretrieve`), so the catalog and the tool pin the same real assets.

**Interfaces:**
- Produces: `fetch_pinned(repo_id: str, filename: str, sha256: str, dest: Path) -> Path` (HF download + SHA verify, atomic); `CATALOG: dict[str, CatalogEntry]` (`CatalogEntry` = dataclass `name, repo_id, filename, sha256, variant, num_keypoints, description`); `resolve_checkpoint(name_or_path: str, cache_dir: Path) -> Path`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_checkpoints.py
import hashlib
import pytest
from pathlib import Path
from hydra_suite.posekit.core.vitpose_checkpoints import (
    fetch_pinned, resolve_checkpoint, CATALOG,
)

def test_resolve_passthrough_local_path(tmp_path):
    f = tmp_path / "mine.pth"; f.write_bytes(b"abc")
    assert resolve_checkpoint(str(f), tmp_path / "cache") == f

def test_fetch_pinned_verifies_sha(tmp_path, monkeypatch):
    payload = b"weights-bytes"
    good = hashlib.sha256(payload).hexdigest()
    src = tmp_path / "hf_cache_file.pth"; src.write_bytes(payload)

    monkeypatch.setattr(
        "hydra_suite.posekit.core.vitpose_checkpoints.hf_hub_download",
        lambda repo_id, filename: str(src),
    )
    dest = tmp_path / "w.pth"
    out = fetch_pinned("repo/x", "w.pth", good, dest)
    assert out.read_bytes() == payload
    # wrong hash must raise and not leave a file behind
    with pytest.raises(ValueError):
        fetch_pinned("repo/x", "w.pth", "0" * 64, tmp_path / "bad.pth")
    assert not (tmp_path / "bad.pth").exists()

def test_catalog_has_coco_b_entry():
    assert "vitpose-b-coco" in CATALOG
    e = CATALOG["vitpose-b-coco"]
    assert e.variant == "B" and e.num_keypoints == 17
    for entry in CATALOG.values():
        assert len(entry.sha256) == 64 and entry.repo_id and entry.filename
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vitpose_checkpoints.py -q`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/posekit/core/vitpose_checkpoints.py
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    repo_id: str
    filename: str
    sha256: str
    variant: str
    num_keypoints: int
    description: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_pinned(repo_id: str, filename: str, sha256: str, dest: Path) -> Path:
    """Download via HF hub (mirrors tools/vitpose/fetch_assets.py) and verify the
    pinned SHA256. Atomic: a hash mismatch leaves no file at `dest`."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and _sha256(dest) == sha256:
        return dest
    src = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(src.read_bytes())
    got = _sha256(tmp)
    if got != sha256:
        os.remove(tmp)
        raise ValueError(
            f"SHA256 mismatch for {repo_id}/{filename}\n  expected {sha256}\n  got      {got}"
        )
    os.replace(tmp, dest)
    return dest


# COCO pins are the real values from tools/vitpose/fetch_assets.py (already
# Spec-1-validated). Add L/H and the AP-10K/APT-36K animal entries the same way:
# a real (repo_id, filename, sha256) plus a backbone-strict load test (below)
# BEFORE the entry ships. If an animal asset cannot be pinned, omit it (COCO-only
# catalog + Browse is the spec's accepted fallback).
CATALOG: dict[str, CatalogEntry] = {
    "vitpose-b-coco": CatalogEntry(
        name="ViTPose-B (COCO)",
        repo_id="nielsr/vitpose-original-checkpoints",
        filename="vitpose-b.pth",
        sha256="2e849e1f1dbb5b87191eda7171f1b16468d5d082a7380e93e94b7ce76a061679",
        variant="B", num_keypoints=17,
        description="ViTPose-B, human COCO-17. General-purpose start.",
    ),
    # "vitpose-l-coco", "vitpose-h-coco", "vitpose-b-ap10k" (animal), ... added likewise.
}


def resolve_checkpoint(name_or_path: str, cache_dir: Path) -> Path:
    if name_or_path in CATALOG:
        e = CATALOG[name_or_path]
        return fetch_pinned(e.repo_id, e.filename, e.sha256, Path(cache_dir) / f"{name_or_path}.pth")
    p = Path(name_or_path)
    if p.exists():
        return p
    raise ValueError(f"not a catalog name or existing file: {name_or_path!r}")
```

> **Implementer note (data, not logic):** `vitpose-b-coco` above carries the real
> pin. For L/H copy the corresponding `nielsr/vitpose-original-checkpoints`
> filenames + SHAs from `tools/vitpose/fetch_assets.py`. For AP-10K/APT-36K, pin
> the upstream animal release and add, per catalog entry, a test that does
> `load_finetune_init(build_finetune_model(e.variant, e.num_keypoints, 0.0), fetched)`
> with no error before shipping the entry.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vitpose_checkpoints.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/posekit/core/vitpose_checkpoints.py tests/test_vitpose_checkpoints.py
git commit -m "feat(vitpose-finetune): base-checkpoint catalog + SHA-pinned downloader/resolver"
```

---

### Task 11: Run orchestration (split, run.json, command, progress parse)

**Files:**
- Create: `src/hydra_suite/posekit/core/vitpose_training.py`
- Test: `tests/test_vitpose_training_orchestration.py`

**Interfaces:**
- Consumes: `RunConfig` (Task 1, imported from the leaf payload — allowed: `posekit` may import `core`); `resolve_checkpoint` (Task 10).
- Produces: `prepare_run(params: dict, run_dir: Path, cache_dir: Path) -> Path` (resolves checkpoint, writes+validates `run.json`, returns its path); `build_training_command(run_json: Path) -> list[str]` (`[sys.executable, "-m", "hydra_suite.core.identity.pose.vitpose.training", "--config", str(run_json)]`); `parse_progress_line(line: str) -> dict | None` (parses `EPOCH …` lines → `{"epoch","train_loss","val_loss","pck@0.05","pck@0.1"}`, else `None`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_training_orchestration.py
import sys
import json
from pathlib import Path
from hydra_suite.posekit.core.vitpose_training import (
    prepare_run, build_training_command, parse_progress_line,
)

def test_prepare_run_writes_valid_config(tmp_path):
    ckpt = tmp_path / "w.pth"; ckpt.write_bytes(b"x")
    run_dir = tmp_path / "run"
    params = dict(
        init_checkpoint=str(ckpt), variant="B", num_keypoints=5,
        dataset_dir=str(tmp_path / "ds"), device="cpu", epochs=10, batch_size=8,
    )
    rj = prepare_run(params, run_dir, cache_dir=tmp_path / "cache")
    d = json.loads(rj.read_text())
    assert d["variant"] == "B" and d["num_keypoints"] == 5
    assert d["output_dir"] == str(run_dir)

def test_build_command_uses_current_interpreter(tmp_path):
    rj = tmp_path / "run.json"
    cmd = build_training_command(rj)
    assert cmd[0] == sys.executable
    assert cmd[1:4] == ["-m", "hydra_suite.core.identity.pose.vitpose.training", "--config"] or \
           cmd[1:3] == ["-m", "hydra_suite.core.identity.pose.vitpose.training"]
    assert str(rj) in cmd

def test_parse_progress():
    line = "EPOCH 4 train_loss=0.00123 val_loss=0.00456 pck@0.05=0.8000 pck@0.1=0.9500"
    r = parse_progress_line(line)
    assert r == {"epoch": 4, "train_loss": 0.00123, "val_loss": 0.00456,
                 "pck@0.05": 0.8, "pck@0.1": 0.95}
    assert parse_progress_line("random log line") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vitpose_training_orchestration.py -q`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/posekit/core/vitpose_training.py
from __future__ import annotations

import re
import sys
from pathlib import Path

from hydra_suite.core.identity.pose.vitpose.training.config import validate_run_config
from hydra_suite.posekit.core.vitpose_checkpoints import resolve_checkpoint

_MODULE = "hydra_suite.core.identity.pose.vitpose.training"
_LINE = re.compile(
    r"^EPOCH (?P<epoch>\d+) train_loss=(?P<tl>[\d.eE+-]+) val_loss=(?P<vl>[\d.eE+-]+) "
    r"pck@0\.05=(?P<p5>[\d.eE+-]+) pck@0\.1=(?P<p1>[\d.eE+-]+)\s*$"
)


def prepare_run(params: dict, run_dir: Path, cache_dir: Path) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_checkpoint(params["init_checkpoint"], cache_dir)
    merged = dict(params)
    merged["init_checkpoint"] = str(resolved)
    merged["output_dir"] = str(run_dir)
    cfg = validate_run_config(merged)  # raises ValueError on bad params
    run_json = run_dir / "run.json"
    cfg.to_json(run_json)
    return run_json


def build_training_command(run_json: Path) -> list[str]:
    return [sys.executable, "-m", _MODULE, "--config", str(run_json)]


def parse_progress_line(line: str) -> dict | None:
    m = _LINE.match(line.strip())
    if not m:
        return None
    return {
        "epoch": int(m["epoch"]),
        "train_loss": float(m["tl"]),
        "val_loss": float(m["vl"]),
        "pck@0.05": float(m["p5"]),
        "pck@0.1": float(m["p1"]),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vitpose_training_orchestration.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/posekit/core/vitpose_training.py tests/test_vitpose_training_orchestration.py
git commit -m "feat(vitpose-finetune): non-Qt orchestration (run.json, subprocess command, progress parse)"
```

---

### Task 12: PoseKit dialog worker + UI wiring

**Files:**
- Modify: `src/hydra_suite/posekit/gui/dialogs/training.py` (add `ViTPoseTrainingWorker`; activate combo entry; add variant + init-checkpoint controls)
- Test: `tests/test_vitpose_training_worker.py`

**Interfaces:**
- Consumes: `prepare_run`, `build_training_command`, `parse_progress_line` (Task 11); `build_coco_keypoints_dataset` (existing). Worker mirrors `TrainingWorker`: signals `log/progress/finished/failed`; `cancel()` → `self._proc.terminate()`; `run()` → `Popen(..., stdout=PIPE, stderr=STDOUT, text=True, bufsize=1, env=os.environ.copy())`, streams lines, drives `progress` off `parse_progress_line`.

**Two codebase conventions to honor (from CLAUDE.md):**
- **Worker base class.** CLAUDE.md prescribes `BaseWorker(QThread)` (`widgets/workers.py`) for background tasks. The *sibling* `TrainingWorker` in this same dialog is still a plain `QObject` (Worker-Pattern Slice 1 is in progress and has not migrated it). **Deliberately mirror the sibling `TrainingWorker` (QObject + explicit `log/progress/finished/failed` signals)** so the two pose workers stay identical and migrate together when Slice 1 reaches this file — do not half-migrate one worker to `BaseWorker` in isolation. Note this choice in the task report so the reviewer sees it is intentional, not an oversight.
- **Paths.** `run_dir` and `cache_dir` must resolve via `hydra_suite.paths` (e.g. `get_models_dir()` for the checkpoint cache; the same training-runs location the YOLO path uses for `run_dir`) — never hardcoded. The dialog computes them and passes them into the worker; the worker and orchestration receive them as arguments (as the test does), staying path-policy-free and testable.

- [ ] **Step 1: Write the failing test** (mock `Popen`; no real training)

```python
# tests/test_vitpose_training_worker.py
import types
from pathlib import Path
import hydra_suite.posekit.gui.dialogs.training as T

class _FakeProc:
    def __init__(self, lines):
        self.stdout = iter(lines)
        self._alive = True
    def poll(self):
        return None if self._alive else 0
    def wait(self):
        self._alive = False
        return 0
    def terminate(self):
        self._alive = False

def test_worker_streams_progress(monkeypatch, tmp_path):
    lines = [
        "EPOCH 0 train_loss=0.5 val_loss=0.4 pck@0.05=0.1 pck@0.1=0.2\n",
        "EPOCH 1 train_loss=0.3 val_loss=0.2 pck@0.05=0.5 pck@0.1=0.7\n",
        "DONE best_pck=0.5 best_epoch=1\n",
    ]
    # stub the dataset build + run prep so the worker does no real IO
    monkeypatch.setattr(T, "build_coco_keypoints_dataset",
                        lambda **kw: {"dataset_dir": tmp_path / "ds", "coco_path": tmp_path / "ds/annotations.json", "labeled_count": 8, "manifest": tmp_path / "m.json"})
    monkeypatch.setattr(T, "prepare_run", lambda params, run_dir, cache_dir: Path(run_dir) / "run.json")
    monkeypatch.setattr(T, "build_training_command", lambda rj: ["true"])
    captured = {}
    monkeypatch.setattr(T.subprocess, "Popen", lambda *a, **k: _FakeProc(lines))

    w = T.ViTPoseTrainingWorker(
        image_paths=[], labels_dir=tmp_path, run_dir=tmp_path / "run",
        cache_dir=tmp_path / "cache", class_names=["a"], keypoint_names=["k0"],
        skeleton_edges=[], variant="B", init_checkpoint="vitpose-b-coco",
        num_keypoints=1, epochs=2, batch=4, device="cpu",
    )
    progresses = []
    w.progress.connect(lambda cur, tot: progresses.append((cur, tot)))
    w.run()
    assert (1, 2) in progresses  # epoch 1 of 2 reported

def test_worker_cancel_terminates(monkeypatch, tmp_path):
    fp = _FakeProc([])
    w = T.ViTPoseTrainingWorker(
        image_paths=[], labels_dir=tmp_path, run_dir=tmp_path / "run",
        cache_dir=tmp_path / "cache", class_names=["a"], keypoint_names=["k0"],
        skeleton_edges=[], variant="B", init_checkpoint="x", num_keypoints=1,
        epochs=1, batch=1, device="cpu",
    )
    w._proc = fp
    w.cancel()
    assert fp.poll() == 0  # terminated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vitpose_training_worker.py -q`
Expected: FAIL (`ViTPoseTrainingWorker` undefined).

- [ ] **Step 3: Write minimal implementation** — add imports at the top of `training.py`:

```python
from hydra_suite.posekit.core.vitpose_training import (
    prepare_run, build_training_command, parse_progress_line,
)
# build_coco_keypoints_dataset is already importable in this module's import block;
# if not, add: from hydra_suite.posekit.core.extensions import build_coco_keypoints_dataset
```

Add the worker class (place beside `TrainingWorker`):

```python
class ViTPoseTrainingWorker(QObject):
    """Runs ViTPose fine-tuning as an in-env subprocess (mirrors TrainingWorker)."""

    log = Signal(str)
    progress = Signal(int, int)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, image_paths, labels_dir, run_dir, cache_dir, class_names,
                 keypoint_names, skeleton_edges, variant, init_checkpoint,
                 num_keypoints, epochs, batch, device):
        super().__init__()
        self.image_paths = list(image_paths)
        self.labels_dir = Path(labels_dir)
        self.run_dir = Path(run_dir)
        self.cache_dir = Path(cache_dir)
        self.class_names = list(class_names)
        self.keypoint_names = list(keypoint_names)
        self.skeleton_edges = list(skeleton_edges)
        self.variant = variant
        self.init_checkpoint = init_checkpoint
        self.num_keypoints = int(num_keypoints)
        self.epochs = int(epochs)
        self.batch = int(batch)
        self.device = device
        self._cancel = False
        self._proc = None

    def cancel(self):
        self._cancel = True
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass

    def run(self):
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.log.emit("Building COCO-keypoints dataset…")
            ds = build_coco_keypoints_dataset(
                image_paths=self.image_paths, labels_dir=self.labels_dir,
                output_dir=self.run_dir / "dataset", class_names=self.class_names,
                keypoint_names=self.keypoint_names, skeleton_edges=self.skeleton_edges,
            )
            params = dict(
                init_checkpoint=self.init_checkpoint, variant=self.variant,
                num_keypoints=self.num_keypoints, dataset_dir=str(ds["dataset_dir"]),
                device=self.device, epochs=self.epochs, batch_size=self.batch,
            )
            run_json = prepare_run(params, self.run_dir, self.cache_dir)
            cmd = build_training_command(run_json)
            self.log.emit(f"Launching: {' '.join(cmd)}")
            self.progress.emit(0, max(1, self.epochs))

            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=os.environ.copy(),
            )
            for line in self._proc.stdout:
                if self._cancel:
                    break
                self.log.emit(line.rstrip())
                prog = parse_progress_line(line)
                if prog is not None:
                    self.progress.emit(prog["epoch"] + 1, self.epochs)
            code = self._proc.wait()
            if self._cancel:
                self.failed.emit("Training cancelled.")
            elif code != 0:
                self.failed.emit(f"Training subprocess exited with code {code}.")
            else:
                self.finished.emit({"run_dir": str(self.run_dir),
                                    "best": str(self.run_dir / "best.pt")})
        except Exception as exc:  # surfaced to the dialog
            self.failed.emit(str(exc))
```

Then wire the UI (in the dialog's control setup): the backend combo already lists `"ViTPose (soon)"` — rename to `"ViTPose"`; on selecting it, show a variant combo (`["B", "L", "H"]`, default `"B"`) and an **editable** init-checkpoint combo populated from `CATALOG` names + a `"Browse…"` sentinel that opens a file dialog. On "Start", when backend == "ViTPose", construct `ViTPoseTrainingWorker` with the project's `class_names/keypoint_names/skeleton_edges`, the selected variant, the checkpoint (catalog key or browsed path), `num_keypoints=len(keypoint_names)`, and the existing epoch/batch/device controls; move it to a `QThread` exactly as `TrainingWorker` is.

```python
# combo activation (locate the existing addItems and replace the label)
self.backend_combo.addItems(["YOLO Pose", "ViTPose", "SLEAP"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vitpose_training_worker.py -q`
Expected: PASS.

- [ ] **Step 5: Manual smoke (GUI import intact) + commit**

```bash
python -c "import hydra_suite.posekit.gui.dialogs.training"
git add src/hydra_suite/posekit/gui/dialogs/training.py tests/test_vitpose_training_worker.py
git commit -m "feat(vitpose-finetune): PoseKit ViTPose training worker + dialog wiring"
```

---

## Final verification (run before finishing the branch)

- [ ] `pytest tests/test_vitpose_finetune_*.py tests/test_vitpose_checkpoints.py tests/test_vitpose_training_*.py -q` — all pass.
- [ ] `pytest tests/ -k "leaf_pure or eager_import" -q` — payload stays leaf-pure.
- [ ] `python -c "from hydra_suite.core.identity.pose.vitpose import build_vitpose"` then confirm `sys.modules` has no `…vitpose.training` key — proves `__init__` does not eager-import training.
- [ ] `python -c "import hydra_suite.posekit.gui.dialogs.training, hydra_suite.posekit.gui.main_window"` — GUI import path intact.
- [ ] `grep -rn "build_vitpose_moe\|dataset_index\|flip_pairs" src/hydra_suite/core/identity/pose/vitpose/training/` — empty (no MoE, no flip leaked in).
- [ ] `make format && make lint` per `CLAUDE.md`.
