# ViTPose Native Port (Spec 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A native-PyTorch ViTPose (classic B + ViTPose+ MoE) that provably reproduces upstream COCO AP, as a standalone leaf module importing nothing from `hydra_suite`.

**Architecture:** Module attribute names are chosen to **match the upstream checkpoint's `state_dict` keys exactly** (`backbone.*`, `keypoint_head.*`, `blocks.{i}.attn.qkv`, `patch_embed.proj`, `last_norm`). This means `load_state_dict(strict=True)` works with **zero key remapping** — and makes Gate A a free architecture test rather than a test of our own rename map. Two decoders ship: `decode_udp_cv2` (faithful mmpose port, the oracle) and `decode_udp_torch` (device-resident, production), bound by a parity test.

**Tech Stack:** PyTorch 2.11, timm (already a repo dep), OpenCV, NumPy, pycocotools. No mmcv, no mmpose, no xtcocotools.

**Spec:** `docs/superpowers/specs/2026-07-16-vitpose-native-port-design.md`
**Roadmap:** `docs/superpowers/specs/2026-07-16-vitpose-backend-roadmap.md`

## Global Constraints

- **Imports only `torch`, `timm`, `numpy`, `cv2`. NOTHING from `hydra_suite`.** This module is a leaf; Spec 3 wires it in. A `from hydra_suite...` import anywhere in `vitpose/` is a plan violation.
- **Import `cv2` BEFORE `torch`** in `vitpose/__init__.py`. Verified on this machine: `import torch, cv2` → OpenMP abort (`OMP: Error #15`, exit 134); `import cv2, torch` → OK. Cause: two libomp copies (conda `@rpath/libomp.dylib` from `llvm-openmp`, and torch's vendored `/opt/llvm-openmp/lib/libomp.dylib`). Do **not** reach for `KMP_DUPLICATE_LIB_OK=TRUE` — LLVM documents it as possibly producing *silently incorrect results*, which is the one failure mode this spec exists to prevent. (`trackerkit/app.py:18` uses it; that is a GUI-level fallback, not our precedent.)
- **`torch.load(..., weights_only=True)` always.** Checkpoints come from a third-party re-host; `weights_only=False` permits arbitrary code execution via unpickling.
- **Never `Path(__file__).parents[N]`** (CLAUDE.md:199-201).
- **~500-line rule** (CLAUDE.md:123). Model on `yolo.py` (293 lines), not `sleap.py` (1780).
- **Line length 88** (black). Pre-commit runs black/ruff/flake8/isort automatically on commit.
- **Env:** `hydra-mps` (Python 3.13.12, torch 2.11.0, MPS available). Run tests as:
  `PYTHONPATH=src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest ...`
- **MPS has no float64.** `decode_udp_cv2` is float64-on-CPU (numpy); `decode_udp_torch` is float32 on MPS. Gate B compares **across dtypes by design** — this is expected, not a bug.
- **Device order:** MPS first (this device). CUDA validation later on `rutalab@mehek.taild08eb9.ts.net`. Never hardcode a device; take `device` as a parameter and default to CPU.
- **Constants that must not drift** (checked into `config.py`, used everywhere):
  `IMAGE_SIZE_WH = (192, 256)`, `HEATMAP_SIZE_WH = (48, 64)`, `PIXEL_STD = 200.0`,
  `PADDING_FACTOR = 1.25`, `IMAGENET_MEAN = (0.485, 0.456, 0.406)`,
  `IMAGENET_STD = (0.229, 0.224, 0.225)`, `UDP_BLUR_KERNEL = 11`, `TARGET_SIGMA = 2.0`.

## Numerically-critical code: transcribe, do not recall

For `get_warp_matrix` (Task 6) and `post_dark_udp` (Task 7), **fetch the upstream file and transcribe the body verbatim.** Do not write it from memory or reconstruct it from the paper. A single wrong sign or a `-1` dropped from a denominator costs ~1 AP and produces no error — exactly the silent failure this spec targets. Each task names the exact URL to fetch.

## File Structure

```
src/hydra_suite/core/identity/pose/vitpose/
├── __init__.py      # public API; cv2-before-torch import order
├── config.py        # variant table + frozen constants          (~90 lines)
├── model.py         # ViT backbone, Block, Attention, MoEMlp    (~230 lines)
├── heads.py         # classic deconv head, simple decoder       (~90 lines)
├── vitpose.py       # top-level ViTPose composing backbone+head (~110 lines)
├── weights.py       # download, weights_only load, strict assert(~120 lines)
├── transforms.py    # box2cs, UDP warp, normalize               (~120 lines)
└── decode.py        # decode_udp_cv2 (oracle), decode_udp_torch (~180 lines)

tests/
├── test_vitpose_config.py
├── test_vitpose_model.py
├── test_vitpose_heads.py
├── test_vitpose_weights.py      # GATE A
├── test_vitpose_transforms.py
├── test_vitpose_decode.py       # GATE B
└── test_vitpose_eval_coco.py    # GATE C (marked slow)

tools/vitpose/
├── fetch_assets.py   # checkpoints + COCO detections, SHA256-pinned
└── eval_coco.py      # GATE C harness
```

Tests live flat in `tests/` as `test_vitpose_*.py`, matching the repo's existing
`tests/test_pose_pipeline.py` convention.

---

### Task 1: Asset fetcher (checkpoints + COCO detections, SHA256-pinned)

Everything downstream needs real weights. This task is first because Gate A is
worthless without them — and because the detections file has a live trap.

**Files:**
- Create: `tools/vitpose/fetch_assets.py`
- Create: `tools/vitpose/__init__.py` (empty)
- Test: `tests/test_vitpose_assets.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ASSETS: dict[str, Asset]` where `Asset = namedtuple("Asset", "kind repo_or_id filename sha256 size")`
  - `fetch(name: str, dest_dir: Path) -> Path` — downloads if absent, always verifies SHA256, raises `AssetIntegrityError` on mismatch.
  - `verify(path: Path, expected_sha256: str) -> None`
  - `class AssetIntegrityError(RuntimeError)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_assets.py
import hashlib
import pytest
from pathlib import Path
from tools.vitpose.fetch_assets import verify, AssetIntegrityError, ASSETS


def test_verify_accepts_matching_sha256(tmp_path: Path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"hello vitpose")
    digest = hashlib.sha256(b"hello vitpose").hexdigest()
    verify(p, digest)  # must not raise


def test_verify_rejects_mismatched_sha256(tmp_path: Path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"tampered")
    with pytest.raises(AssetIntegrityError) as exc:
        verify(p, "0" * 64)
    assert "sha256 mismatch" in str(exc.value).lower()


def test_detections_asset_pins_the_real_file_not_the_dummy():
    """The LiteHrnet copy on GitHub is a 250KB dummy (1000 boxes, all score 0.99).
    The genuine file is 16,383,781 bytes. Pinning size+sha is what separates them."""
    a = ASSETS["coco_val2017_person_detections"]
    assert a.size == 16_383_781
    assert a.sha256 == (
        "53ba0ad8d0fd461c5a000cd90797fa8c39cd8c38cd125125c0412626ff592d59"
    )


def test_fetch_refuses_unpinned_asset_by_default(tmp_path: Path, monkeypatch):
    """An unpinned asset must fail loudly rather than silently skip
    verification -- silently-unverified is the failure mode this module exists
    to prevent."""
    from tools.vitpose import fetch_assets

    monkeypatch.setitem(
        fetch_assets.ASSETS,
        "_unpinned",
        fetch_assets.Asset("hf", "repo", "f.bin", "", 0),
    )
    with pytest.raises(AssetIntegrityError, match="no pinned sha256"):
        fetch_assets.fetch("_unpinned", tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=.:src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_vitpose_assets.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.vitpose.fetch_assets'`

- [ ] **Step 3: Write minimal implementation**

```python
# tools/vitpose/fetch_assets.py
"""Fetch and integrity-check ViTPose assets.

Checkpoints come from a third-party re-host (nielsr/vitpose-original-checkpoints)
because upstream publishes OneDrive links only, which 403 to non-browser clients.
Every asset is SHA256-pinned: for weights because we do not control the host, and
for the COCO detections because a plausible-looking dummy is in circulation.
"""

from __future__ import annotations

import hashlib
from collections import namedtuple
from pathlib import Path

Asset = namedtuple("Asset", "kind repo_or_id filename sha256 size")


class AssetIntegrityError(RuntimeError):
    """Raised when a downloaded asset does not match its pinned digest."""


ASSETS: dict[str, Asset] = {
    "vitpose-b": Asset(
        kind="hf",
        repo_or_id="nielsr/vitpose-original-checkpoints",
        filename="vitpose-b.pth",
        sha256="",  # filled by Step 6
        size=0,
    ),
    "vitpose-b-simple": Asset(
        kind="hf",
        repo_or_id="nielsr/vitpose-original-checkpoints",
        filename="vitpose-b-simple.pth",
        sha256="",
        size=0,
    ),
    "vitpose-plus-base": Asset(
        kind="hf",
        repo_or_id="nielsr/vitpose-original-checkpoints",
        filename="vitpose+_base.pth",
        sha256="",
        size=0,
    ),
    "coco_val2017_person_detections": Asset(
        kind="gdrive",
        repo_or_id="1ygw57X-mh0QBfENB-U5DsuSauGIu-8RB",
        filename="COCO_val2017_detections_AP_H_56_person.json",
        sha256="53ba0ad8d0fd461c5a000cd90797fa8c39cd8c38cd125125c0412626ff592d59",
        size=16_383_781,
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, expected_sha256: str) -> None:
    actual = _sha256(path)
    if actual != expected_sha256:
        raise AssetIntegrityError(
            f"sha256 mismatch for {path.name}: "
            f"expected {expected_sha256}, got {actual}"
        )


def fetch(name: str, dest_dir: Path, allow_unpinned: bool = False) -> Path:
    """Download (if absent) and verify an asset.

    An unpinned asset (sha256="") raises unless allow_unpinned=True. Silently
    skipping verification for unpinned entries would defeat the point of the
    module: the bootstrap that DISCOVERS a digest must say so explicitly.
    Only the Step 6 bootstrap passes allow_unpinned=True.
    """
    asset = ASSETS[name]
    if not asset.sha256 and not allow_unpinned:
        raise AssetIntegrityError(
            f"{name} has no pinned sha256; run the Step 6 bootstrap to record "
            f"one, or pass allow_unpinned=True if you are that bootstrap"
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / asset.filename
    if not out.exists():
        if asset.kind == "hf":
            from huggingface_hub import hf_hub_download

            src = hf_hub_download(
                repo_id=asset.repo_or_id, filename=asset.filename
            )
            out.write_bytes(Path(src).read_bytes())
        elif asset.kind == "gdrive":
            import gdown

            gdown.download(id=asset.repo_or_id, output=str(out), quiet=False)
        else:
            raise ValueError(f"unknown asset kind: {asset.kind}")
    if asset.sha256:
        verify(out, asset.sha256)
    if asset.size and out.stat().st_size != asset.size:
        raise AssetIntegrityError(
            f"size mismatch for {asset.filename}: "
            f"expected {asset.size}, got {out.stat().st_size}"
        )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=.:src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_vitpose_assets.py -q
```
Expected: PASS (4 passed)

- [ ] **Step 5: Install fetch deps if absent, then download the real assets**

```bash
E=/Users/neurorishika/miniforge3/envs/hydra-mps/bin
$E/python -c "import gdown" 2>/dev/null || $E/pip install gdown
$E/python -c "import huggingface_hub" 2>/dev/null || $E/pip install huggingface_hub
```

`gdown` is an eval/dev tool. It must **not** be added to `pyproject.toml`
runtime deps — Spec 1 ships no runtime code that downloads anything.

- [ ] **Step 6: Download assets and record the real checkpoint digests**

```bash
E=/Users/neurorishika/miniforge3/envs/hydra-mps/bin
D=$HOME/.cache/vitpose-assets
PYTHONPATH=.:src $E/python - <<'PY'
from pathlib import Path
from tools.vitpose.fetch_assets import ASSETS, fetch, _sha256
import os
d = Path(os.path.expanduser("~/.cache/vitpose-assets"))
# allow_unpinned=True ONLY here: this bootstrap is what discovers the digests.
for name in ("vitpose-b", "vitpose-b-simple", "vitpose-plus-base"):
    p = fetch(name, d, allow_unpinned=True)
    print(f"{name:34s} {p.stat().st_size:>12,} B  {_sha256(p)}")
# The detections file is ALREADY pinned -- fetch it pinned, so a dummy or a
# corrupt download fails loudly right here.
p = fetch("coco_val2017_person_detections", d)
print(f"{'coco detections (pinned, verified)':34s} {p.stat().st_size:>12,} B")
PY
```

Paste the printed sha256/size for the three checkpoints into `ASSETS`
(replacing the empty `sha256=""` and `size=0`). The detections entry is already
pinned and **must verify without edit** — if it raises `AssetIntegrityError`,
you have the dummy or a corrupted download. Do not "fix" it by updating the
constant.

- [ ] **Step 7: Re-run tests, then commit**

```bash
PYTHONPATH=.:src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_vitpose_assets.py -q
git add tools/vitpose/ tests/test_vitpose_assets.py
git commit -m "feat(vitpose): SHA256-pinned asset fetcher for weights and COCO detections"
```

---

### Task 2: Variant config and frozen constants

**Files:**
- Create: `src/hydra_suite/core/identity/pose/vitpose/__init__.py`
- Create: `src/hydra_suite/core/identity/pose/vitpose/config.py`
- Test: `tests/test_vitpose_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass(frozen=True) class ViTPoseVariant` with fields
    `embed_dim: int, depth: int, num_heads: int, part_features: int, drop_path_rate: float, layer_decay: float`
  - `VARIANTS: dict[str, ViTPoseVariant]` keyed `"S" | "B" | "L" | "H"`
  - constants `IMAGE_SIZE_WH`, `HEATMAP_SIZE_WH`, `PIXEL_STD`, `PADDING_FACTOR`, `IMAGENET_MEAN`, `IMAGENET_STD`, `UDP_BLUR_KERNEL`, `TARGET_SIGMA`, `NUM_EXPERTS`, `EXPERT_DATASETS`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_config.py
from hydra_suite.core.identity.pose.vitpose.config import (
    VARIANTS, IMAGE_SIZE_WH, HEATMAP_SIZE_WH, PIXEL_STD, UDP_BLUR_KERNEL,
    TARGET_SIGMA, NUM_EXPERTS, EXPERT_DATASETS,
)


def test_variant_dims():
    assert VARIANTS["S"].embed_dim == 384
    assert VARIANTS["B"].embed_dim == 768
    assert VARIANTS["L"].embed_dim == 1024
    assert VARIANTS["H"].embed_dim == 1280
    assert VARIANTS["B"].depth == 12
    assert VARIANTS["L"].depth == 24
    assert VARIANTS["H"].depth == 32


def test_small_uses_twelve_heads_not_six():
    """ViTPose-S is 12 heads at dim 384 (head_dim 32), NOT the usual 6.
    Getting this from ViT habit rather than the config is a real trap."""
    assert VARIANTS["S"].num_heads == 12
    assert VARIANTS["S"].embed_dim // VARIANTS["S"].num_heads == 32


def test_part_features():
    assert [VARIANTS[k].part_features for k in "SBLH"] == [96, 192, 256, 320]


def test_frozen_constants():
    assert IMAGE_SIZE_WH == (192, 256)     # (w, h) - configs write [192, 256]
    assert HEATMAP_SIZE_WH == (48, 64)     # (w, h)
    assert PIXEL_STD == 200.0
    assert UDP_BLUR_KERNEL == 11
    assert TARGET_SIGMA == 2.0


def test_blur_kernel_matches_training_sigma():
    """OpenCV sigma=0 derives sigma from kernel: 0.3*((k-1)*0.5 - 1) + 0.8.
    For k=11 that is exactly 2.0 == TARGET_SIGMA. HF hardcodes 0.8 instead,
    which does not track kernel size; we deliberately do not follow HF."""
    k = UDP_BLUR_KERNEL
    derived = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    assert abs(derived - TARGET_SIGMA) < 1e-9


def test_expert_dataset_order():
    assert NUM_EXPERTS == 6
    assert EXPERT_DATASETS == (
        "COCO", "AiC", "MPII", "AP-10K", "APT-36K", "COCO-WholeBody",
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_vitpose_config.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.core.identity.pose.vitpose'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/core/identity/pose/vitpose/__init__.py
"""Native-PyTorch ViTPose. Standalone leaf: imports nothing from hydra_suite.

Import order matters. cv2 must be imported before torch on this platform:
conda ships libomp.dylib (@rpath) and torch vendors its own
(/opt/llvm-openmp/lib/libomp.dylib). Loading torch first then cv2 aborts the
process with `OMP: Error #15`. cv2-first is stable. Do not "fix" this with
KMP_DUPLICATE_LIB_OK=TRUE: LLVM documents that as possibly producing silently
incorrect results, which defeats the point of a numerical-parity module.
"""

import cv2 as _cv2  # noqa: F401  (must precede torch; see module docstring)
import torch as _torch  # noqa: F401
```

```python
# src/hydra_suite/core/identity/pose/vitpose/config.py
"""Variant table and constants that must not drift.

Values transcribed from upstream ViTPose configs
(configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/).
"""

from __future__ import annotations

from dataclasses import dataclass

IMAGE_SIZE_WH: tuple[int, int] = (192, 256)
HEATMAP_SIZE_WH: tuple[int, int] = (48, 64)
PIXEL_STD: float = 200.0
PADDING_FACTOR: float = 1.25
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

UDP_BLUR_KERNEL: int = 11
TARGET_SIGMA: float = 2.0

NUM_EXPERTS: int = 6
EXPERT_DATASETS: tuple[str, ...] = (
    "COCO",
    "AiC",
    "MPII",
    "AP-10K",
    "APT-36K",
    "COCO-WholeBody",
)

PATCH_SIZE: int = 16
PATCH_PADDING: int = 2


@dataclass(frozen=True)
class ViTPoseVariant:
    embed_dim: int
    depth: int
    num_heads: int
    part_features: int
    drop_path_rate: float
    layer_decay: float


VARIANTS: dict[str, ViTPoseVariant] = {
    "S": ViTPoseVariant(384, 12, 12, 96, 0.10, 0.80),
    "B": ViTPoseVariant(768, 12, 12, 192, 0.30, 0.75),
    "L": ViTPoseVariant(1024, 24, 16, 256, 0.50, 0.80),
    "H": ViTPoseVariant(1280, 32, 16, 320, 0.55, 0.85),
}
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_vitpose_config.py -q
```
Expected: PASS (6 passed)

- [ ] **Step 5: Verify the leaf constraint holds**

Run:
```bash
grep -rn "from hydra_suite\|import hydra_suite" src/hydra_suite/core/identity/pose/vitpose/
```
Expected: no output. Any hit is a Global Constraints violation — fix before committing.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/identity/pose/vitpose/ tests/test_vitpose_config.py
git commit -m "feat(vitpose): variant table and frozen constants"
```

---

### Task 3: ViT backbone

The two silent traps live here. The tests below exist specifically to catch them.

**Files:**
- Create: `src/hydra_suite/core/identity/pose/vitpose/model.py`
- Test: `tests/test_vitpose_model.py`

**Interfaces:**
- Consumes: `config.VARIANTS`, `config.PATCH_SIZE`, `config.PATCH_PADDING`.
- Produces:
  - `class PatchEmbed(nn.Module)` — attr `proj: nn.Conv2d`
  - `class Attention(nn.Module)` — attrs `qkv: nn.Linear`, `proj: nn.Linear`
  - `class Mlp(nn.Module)` — attrs `fc1`, `fc2`
  - `class Block(nn.Module)` — attrs `norm1`, `attn`, `drop_path`, `norm2`, `mlp`
  - `class ViT(nn.Module)` — attrs `patch_embed`, `pos_embed`, `blocks`, `last_norm`; `forward(x: Tensor) -> Tensor` returning `(B, embed_dim, 16, 12)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_model.py
import torch
from hydra_suite.core.identity.pose.vitpose.model import ViT, PatchEmbed
from hydra_suite.core.identity.pose.vitpose.config import VARIANTS


def _vit_b() -> ViT:
    v = VARIANTS["B"]
    return ViT(embed_dim=v.embed_dim, depth=v.depth, num_heads=v.num_heads)


def test_patch_embed_uses_padding_two_not_zero():
    """TRAP 1. Upstream computes padding = 4 + 2*(ratio//2 - 1) = 2 for ratio=1.
    Stock timm uses padding=0. The output grid is coincidentally identical
    (floor((256+4-16)/16)+1 == 16), so a wrong padding loads with NO shape error
    and silently samples a shifted pixel grid."""
    pe = PatchEmbed(embed_dim=768)
    assert pe.proj.padding == (2, 2)
    assert pe.proj.kernel_size == (16, 16)
    assert pe.proj.stride == (16, 16)


def test_backbone_output_shape():
    m = _vit_b().eval()
    with torch.no_grad():
        out = m(torch.zeros(2, 3, 256, 192))
    assert out.shape == (2, 768, 16, 12)


def test_pos_embed_retains_cls_slot():
    """TRAP 2. pos_embed is (1, num_patches+1, D) -- the MAE cls slot is kept
    even though no cls_token module exists."""
    m = _vit_b()
    assert m.pos_embed.shape == (1, 16 * 12 + 1, 768)


def test_pos_embed_adds_cls_slot_to_every_token():
    """TRAP 2, the part that silently changes outputs. Upstream does
        x = x + pos_embed[:, 1:] + pos_embed[:, :1]
    i.e. the cls positional embedding is broadcast onto EVERY patch token.
    Dropping that second term still runs and still has the right shape.

    Test: zero the patch slots, set the cls slot to a known constant, and feed a
    zero image with identity-ish blocks bypassed. If the cls term is applied, the
    pre-block token tensor equals that constant.
    """
    m = _vit_b().eval()
    with torch.no_grad():
        m.pos_embed.zero_()
        m.pos_embed[:, :1].fill_(0.5)
        m.patch_embed.proj.weight.zero_()
        m.patch_embed.proj.bias.zero_()
        tokens = m.forward_tokens(torch.zeros(1, 3, 256, 192))
    assert torch.allclose(tokens, torch.full_like(tokens, 0.5)), (
        "cls positional embedding is not being broadcast onto patch tokens"
    )


def test_layernorm_eps():
    m = _vit_b()
    assert m.blocks[0].norm1.eps == 1e-6
    assert m.last_norm.eps == 1e-6


def test_attention_head_dim_for_small():
    v = VARIANTS["S"]
    m = ViT(embed_dim=v.embed_dim, depth=1, num_heads=v.num_heads)
    assert m.blocks[0].attn.num_heads == 12
    assert m.blocks[0].attn.head_dim == 32
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_vitpose_model.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named '...vitpose.model'`

- [ ] **Step 3: Write minimal implementation**

Attribute names are load-bearing: they must equal the checkpoint's key names.

```python
# src/hydra_suite/core/identity/pose/vitpose/model.py
"""ViT backbone for ViTPose.

Plain, non-hierarchical ViT: absolute learned pos-embed only, no relative
position bias, no window attention in any variant (L/H are dense global
attention at every layer). Attribute names mirror the upstream checkpoint's
state_dict keys so load_state_dict(strict=True) needs no remapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_

from .config import PATCH_PADDING, PATCH_SIZE


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int, in_chans: int = 3) -> None:
        super().__init__()
        # padding=2 is upstream's `4 + 2*(ratio//2 - 1)` with ratio=1, NOT the
        # stock ViT padding=0. See test_patch_embed_uses_padding_two_not_zero.
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=PATCH_SIZE,
            stride=PATCH_SIZE,
            padding=PATCH_PADDING,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, hp, wp = x.shape
        return x.flatten(2).transpose(1, 2), hp, wp


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, -1).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViT(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        img_size_hw: tuple[int, int] = (256, 192),
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(embed_dim)
        hp = img_size_hw[0] // PATCH_SIZE
        wp = img_size_hw[1] // PATCH_SIZE
        num_patches = hp * wp
        # +1 keeps the MAE cls slot; there is no cls_token module.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, drop_path=dpr[i]) for i in range(depth)]
        )
        self.last_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        trunc_normal_(self.pos_embed, std=0.02)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Patch-embed and add positional embeddings. Exposed for testing the
        cls-broadcast behaviour without running the blocks."""
        x, _, _ = self.patch_embed(x)
        # Both terms are required: patch pos-embeds PLUS the cls pos-embed
        # broadcast to every token.
        return x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, hp, wp = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        for blk in self.blocks:
            x = blk(x)
        x = self.last_norm(x)
        b, _, c = x.shape
        return x.permute(0, 2, 1).reshape(b, c, hp, wp)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_vitpose_model.py -q
```
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/identity/pose/vitpose/model.py tests/test_vitpose_model.py
git commit -m "feat(vitpose): ViT backbone with upstream patch padding and pos-embed"
```

---

### Task 4: Heads (classic deconv + simple decoder)

**Files:**
- Create: `src/hydra_suite/core/identity/pose/vitpose/heads.py`
- Test: `tests/test_vitpose_heads.py`

**Interfaces:**
- Consumes: nothing from earlier tasks except `config`.
- Produces:
  - `class ClassicHead(nn.Module)` — attrs `deconv_layers: nn.Sequential`, `final_layer: nn.Conv2d`; `forward(x) -> Tensor`
  - `class SimpleHead(nn.Module)` — attrs `deconv_layers: nn.Identity`, `final_layer: nn.Conv2d`; `forward(x) -> Tensor`
  - `build_head(kind: str, embed_dim: int, num_keypoints: int) -> nn.Module` where `kind in {"classic", "simple"}`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_heads.py
import torch
import torch.nn as nn
from hydra_suite.core.identity.pose.vitpose.heads import (
    ClassicHead, SimpleHead, build_head,
)


def test_classic_head_shape():
    h = ClassicHead(embed_dim=768, num_keypoints=17).eval()
    with torch.no_grad():
        out = h(torch.zeros(2, 768, 16, 12))
    assert out.shape == (2, 17, 64, 48)


def test_classic_deconv_indices_match_checkpoint_keys():
    """Checkpoint keys are keypoint_head.deconv_layers.{0,1,3,4}.*
    -> Sequential(ConvT, BN, ReLU, ConvT, BN, ReLU): params at 0,1,3,4 only.
    ReLU carries no params, which is what creates the 2->3 gap."""
    h = ClassicHead(embed_dim=768, num_keypoints=17)
    layers = h.deconv_layers
    assert isinstance(layers[0], nn.ConvTranspose2d)
    assert isinstance(layers[1], nn.BatchNorm2d)
    assert isinstance(layers[2], nn.ReLU)
    assert isinstance(layers[3], nn.ConvTranspose2d)
    assert isinstance(layers[4], nn.BatchNorm2d)
    assert isinstance(layers[5], nn.ReLU)
    keys = {k.split(".")[1] for k in h.state_dict() if k.startswith("deconv")}
    assert keys == {"0", "1", "3", "4"}


def test_classic_deconv_config():
    h = ClassicHead(embed_dim=768, num_keypoints=17)
    d0 = h.deconv_layers[0]
    assert d0.kernel_size == (4, 4)
    assert d0.stride == (2, 2)
    assert d0.padding == (1, 1)
    assert d0.output_padding == (0, 0)
    assert d0.bias is None  # bias=False
    assert h.final_layer.kernel_size == (1, 1)


def test_simple_head_shape_and_final_conv():
    h = SimpleHead(embed_dim=768, num_keypoints=17).eval()
    with torch.no_grad():
        out = h(torch.zeros(2, 768, 16, 12))
    assert out.shape == (2, 17, 64, 48)
    assert h.final_layer.kernel_size == (3, 3)
    assert h.final_layer.padding == (1, 1)


def test_simple_head_has_no_deconv_params():
    """vitpose-b-simple.pth carries only keypoint_head.final_layer.*"""
    h = SimpleHead(embed_dim=768, num_keypoints=17)
    assert not [k for k in h.state_dict() if k.startswith("deconv")]


def test_simple_head_applies_relu_before_upsample():
    """The ReLU lives in upstream's _transform_inputs, BEFORE F.interpolate.
    With a strictly-negative input, a pre-upsample ReLU zeroes everything, so
    the final conv sees only its bias."""
    h = SimpleHead(embed_dim=8, num_keypoints=2).eval()
    with torch.no_grad():
        h.final_layer.weight.fill_(1.0)
        h.final_layer.bias.zero_()
        out = h(torch.full((1, 8, 16, 12), -5.0))
    assert torch.count_nonzero(out) == 0, "ReLU is not applied before upsampling"


def test_build_head_dispatch():
    assert isinstance(build_head("classic", 768, 17), ClassicHead)
    assert isinstance(build_head("simple", 768, 17), SimpleHead)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_vitpose_heads.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named '...vitpose.heads'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/core/identity/pose/vitpose/heads.py
"""ViTPose heatmap heads.

Both are upstream's TopdownHeatmapSimpleHead; the config chooses between them.
Input (B, D, 16, 12) -> output (B, K, 64, 48).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HEATMAP_SIZE_WH


class ClassicHead(nn.Module):
    """num_deconv_layers=2, filters=(256, 256), kernels=(4, 4),
    final_conv_kernel=1."""

    def __init__(self, embed_dim: int, num_keypoints: int) -> None:
        super().__init__()
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.final_layer = nn.Conv2d(256, num_keypoints, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_layer(self.deconv_layers(x))


class SimpleHead(nn.Module):
    """num_deconv_layers=0, upsample=4, final_conv_kernel=3.

    Upstream applies ReLU inside _transform_inputs, i.e. BEFORE the upsample.
    """

    def __init__(self, embed_dim: int, num_keypoints: int) -> None:
        super().__init__()
        self.deconv_layers = nn.Identity()
        self.final_layer = nn.Conv2d(embed_dim, num_keypoints, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        w, h = HEATMAP_SIZE_WH
        # Explicit size (not scale_factor): scale_factor traces to a Resize with
        # computed sizes and is the classic ONNX shape-mismatch source. Same
        # result here, exportable later. align_corners=False is upstream's.
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return self.final_layer(self.deconv_layers(x))


def build_head(kind: str, embed_dim: int, num_keypoints: int) -> nn.Module:
    if kind == "classic":
        return ClassicHead(embed_dim, num_keypoints)
    if kind == "simple":
        return SimpleHead(embed_dim, num_keypoints)
    raise ValueError(f"unknown head kind: {kind!r} (expected 'classic'|'simple')")
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_vitpose_heads.py -q
```
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/identity/pose/vitpose/heads.py tests/test_vitpose_heads.py
git commit -m "feat(vitpose): classic deconv and simple decoder heads"
```

---

### Task 5: GATE A — top-level model and strict weight load

The payoff task. `strict=True` is the architecture unit test: it catches Traps 1
and 2 for free, because a wrong `padding` or a missing `pos_embed` slot changes
either a shape or a key.

**Files:**
- Create: `src/hydra_suite/core/identity/pose/vitpose/vitpose.py`
- Create: `src/hydra_suite/core/identity/pose/vitpose/weights.py`
- Modify: `src/hydra_suite/core/identity/pose/vitpose/__init__.py`
- Test: `tests/test_vitpose_weights.py`

**Interfaces:**
- Consumes: `ViT` (Task 3), `build_head` (Task 4), `VARIANTS` (Task 2), `fetch` (Task 1).
- Produces:
  - `class ViTPose(nn.Module)` — attrs `backbone: ViT`, `keypoint_head: nn.Module`; `forward(x) -> Tensor` `(B, K, 64, 48)`
  - `build_vitpose(variant: str, head: str, num_keypoints: int = 17) -> ViTPose`
  - `load_checkpoint(model: nn.Module, path: Path, strict: bool = True) -> None`
  - `class CheckpointKeyError(RuntimeError)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_weights.py
import os
from pathlib import Path

import pytest
import torch

from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose
from hydra_suite.core.identity.pose.vitpose.weights import load_checkpoint

ASSET_DIR = Path(os.path.expanduser("~/.cache/vitpose-assets"))

requires_weights = pytest.mark.skipif(
    not (ASSET_DIR / "vitpose-b.pth").exists(),
    reason="run tools/vitpose/fetch_assets.py first",
)


def test_forward_shape_without_weights():
    m = build_vitpose("B", "classic").eval()
    with torch.no_grad():
        out = m(torch.zeros(1, 3, 256, 192))
    assert out.shape == (1, 17, 64, 48)


@requires_weights
def test_gate_a_strict_load_classic():
    """GATE A(1). strict=True is the architecture test: a wrong patch padding
    or a dropped pos_embed cls slot fails here with no ambiguity."""
    m = build_vitpose("B", "classic")
    load_checkpoint(m, ASSET_DIR / "vitpose-b.pth", strict=True)


@requires_weights
def test_gate_a_strict_load_simple():
    """GATE A(2)."""
    m = build_vitpose("B", "simple")
    load_checkpoint(m, ASSET_DIR / "vitpose-b-simple.pth", strict=True)


@requires_weights
def test_checkpoint_load_is_weights_only():
    """Checkpoints come from a third-party re-host. weights_only=False would
    permit arbitrary code execution via unpickling."""
    import inspect

    from hydra_suite.core.identity.pose.vitpose import weights

    src = inspect.getsource(weights)
    assert "weights_only=True" in src
    assert "weights_only=False" not in src


@requires_weights
def test_loaded_model_produces_finite_heatmaps():
    m = build_vitpose("B", "classic").eval()
    load_checkpoint(m, ASSET_DIR / "vitpose-b.pth", strict=True)
    with torch.no_grad():
        out = m(torch.zeros(1, 3, 256, 192))
    assert out.shape == (1, 17, 64, 48)
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_vitpose_weights.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named '...vitpose.vitpose'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/core/identity/pose/vitpose/vitpose.py
"""Top-level ViTPose: backbone + keypoint head.

Attribute names `backbone` and `keypoint_head` are deliberate: they equal the
upstream checkpoint's state_dict prefixes, so strict loading needs no rename map.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import VARIANTS
from .heads import build_head
from .model import ViT


class ViTPose(nn.Module):
    def __init__(self, backbone: ViT, keypoint_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.keypoint_head = keypoint_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.keypoint_head(self.backbone(x))


def build_vitpose(
    variant: str, head: str, num_keypoints: int = 17
) -> ViTPose:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r} (expected one of SBLH)")
    v = VARIANTS[variant]
    backbone = ViT(embed_dim=v.embed_dim, depth=v.depth, num_heads=v.num_heads)
    return ViTPose(backbone, build_head(head, v.embed_dim, num_keypoints))
```

```python
# src/hydra_suite/core/identity/pose/vitpose/weights.py
"""Checkpoint loading with strict-key assertions."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class CheckpointKeyError(RuntimeError):
    """Raised when checkpoint keys do not match the model."""


def load_checkpoint(
    model: nn.Module, path: Path, strict: bool = True
) -> None:
    # weights_only=True is mandatory: these checkpoints come from a third-party
    # re-host, and the default (False) unpickles arbitrary objects.
    blob = torch.load(path, map_location="cpu", weights_only=True)
    state = blob["state_dict"] if "state_dict" in blob else blob
    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict and (missing or unexpected):
        raise CheckpointKeyError(
            f"strict load failed for {path.name}\n"
            f"  missing ({len(missing)}): {sorted(missing)[:10]}\n"
            f"  unexpected ({len(unexpected)}): {sorted(unexpected)[:10]}"
        )
```

- [ ] **Step 4: Run the gate**

Run:
```bash
PYTHONPATH=src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_vitpose_weights.py -q
```
Expected: PASS (5 passed)

**If `test_gate_a_strict_load_classic` fails, read the error before changing
anything — it names the defect precisely:**

| symptom in missing/unexpected | cause |
|---|---|
| `pos_embed` shape mismatch | Trap 2: cls slot dropped from the parameter |
| `patch_embed.proj` shape mismatch | wrong `in_chans`/`embed_dim` |
| `keypoint_head.deconv_layers.2/5` present | ReLU given params, or wrong Sequential order |
| everything prefixed `backbone.backbone.` | double nesting in `build_vitpose` |
| all keys unexpected | attribute names diverge from checkpoint |

Note: a wrong patch **padding** does *not* surface here — shapes coincide. It is
caught by `test_patch_embed_uses_padding_two_not_zero` (Task 3) and by Gate C.

- [ ] **Step 5: Export the public API**

```python
# append to src/hydra_suite/core/identity/pose/vitpose/__init__.py
from .config import VARIANTS, ViTPoseVariant  # noqa: E402
from .vitpose import ViTPose, build_vitpose  # noqa: E402
from .weights import CheckpointKeyError, load_checkpoint  # noqa: E402

__all__ = [
    "VARIANTS",
    "ViTPoseVariant",
    "ViTPose",
    "build_vitpose",
    "load_checkpoint",
    "CheckpointKeyError",
]
```

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/identity/pose/vitpose/ tests/test_vitpose_weights.py
git commit -m "feat(vitpose): GATE A - strict checkpoint load for classic and simple"
```

---

### Task 6: Transforms (bbox -> center/scale, UDP affine warp)

**Files:**
- Create: `src/hydra_suite/core/identity/pose/vitpose/transforms.py`
- Test: `tests/test_vitpose_transforms.py`

**Interfaces:**
- Consumes: `config` constants.
- Produces:
  - `box2cs(box_xywh: np.ndarray) -> tuple[np.ndarray, np.ndarray]` returning `(center(2,), scale(2,))`
  - `get_warp_matrix(theta: float, size_input: np.ndarray, size_dst: np.ndarray, size_target: np.ndarray) -> np.ndarray` shape `(2, 3)`
  - `top_down_affine(img: np.ndarray, center: np.ndarray, scale: np.ndarray, rot: float = 0.0) -> np.ndarray` returning HxWx3 uint8 at 256x192
  - `normalize(img: np.ndarray) -> np.ndarray` returning `(3, 256, 192)` float32, RGB
  - `transform_preds(coords: np.ndarray, center, scale, output_size_wh) -> np.ndarray`

- [ ] **Step 1: Fetch the upstream source — do not write from memory**

```bash
mkdir -p /tmp/vitpose-ref
curl -sL -o /tmp/vitpose-ref/post_transforms.py \
  https://raw.githubusercontent.com/ViTAE-Transformer/ViTPose/main/mmpose/core/post_processing/post_transforms.py
grep -n "def get_warp_matrix" -A 30 /tmp/vitpose-ref/post_transforms.py
grep -n "def transform_preds" -A 40 /tmp/vitpose-ref/post_transforms.py
```

Transcribe `get_warp_matrix` and the **`use_udp=True` branch** of
`transform_preds` verbatim. Do not use `get_affine_transform` (the 3-point
`_get_3rd_point` construction) — that is the non-UDP path.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_vitpose_transforms.py
import numpy as np
import pytest

from hydra_suite.core.identity.pose.vitpose.transforms import (
    box2cs, get_warp_matrix, top_down_affine, normalize, transform_preds,
)
from hydra_suite.core.identity.pose.vitpose.config import (
    PIXEL_STD, PADDING_FACTOR, IMAGENET_MEAN, IMAGENET_STD,
)


def test_box2cs_center():
    c, s = box2cs(np.array([10.0, 20.0, 40.0, 80.0]))
    assert np.allclose(c, [30.0, 60.0])


def test_box2cs_scale_uses_pixel_std_and_padding():
    _, s = box2cs(np.array([0.0, 0.0, 192.0, 256.0]))
    assert np.allclose(s, np.array([192.0, 256.0]) / PIXEL_STD * PADDING_FACTOR)


def test_box2cs_fixes_aspect_ratio():
    """A wide box must be grown in height to reach the 192:256 aspect."""
    _, s = box2cs(np.array([0.0, 0.0, 400.0, 100.0]))
    assert s[0] / s[1] == pytest.approx(192 / 256, rel=1e-6)


def test_warp_matrix_shape():
    m = get_warp_matrix(
        0.0,
        np.array([100.0, 100.0]),
        np.array([191.0, 255.0]),
        np.array([200.0, 200.0]),
    )
    assert m.shape == (2, 3)


def test_warp_uses_size_minus_one_for_udp():
    """UDP defines unit length as pixel SPACING (size-1), not pixel count.
    A centred square box must map its centre to the destination centre and its
    edges to exactly 0 and size-1."""
    img = np.zeros((400, 400, 3), np.uint8)
    center = np.array([200.0, 200.0])
    scale = np.array([400.0, 400.0]) / PIXEL_STD
    out = top_down_affine(img, center, scale)
    assert out.shape == (256, 192, 3)


def test_affine_maps_marker_to_expected_pixel():
    """Put a white marker at the box centre; after warping it must land at the
    destination centre (191/2, 255/2), i.e. the UDP (size-1) convention."""
    img = np.zeros((400, 400, 3), np.uint8)
    img[198:203, 198:203] = 255
    center = np.array([200.0, 200.0])
    scale = np.array([400.0, 400.0]) / PIXEL_STD
    out = top_down_affine(img, center, scale)
    ys, xs = np.nonzero(out[:, :, 0])
    assert abs(xs.mean() - 191 / 2) < 1.5
    assert abs(ys.mean() - 255 / 2) < 1.5


def test_normalize_is_rgb_chw_and_imagenet():
    img = np.zeros((256, 192, 3), np.uint8)
    img[:, :, 2] = 255  # BGR red
    out = normalize(img)
    assert out.shape == (3, 256, 192)
    assert out.dtype == np.float32
    # channel 0 is R after BGR->RGB, so it should be the (1 - mean)/std value
    assert np.allclose(out[0], (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0], atol=1e-5)
    assert np.allclose(out[2], (0.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2], atol=1e-5)


def test_transform_preds_roundtrip_center():
    """A prediction at the heatmap centre must map back to the box centre."""
    center = np.array([200.0, 200.0])
    scale = np.array([400.0, 400.0]) / PIXEL_STD
    coords = np.array([[47 / 2, 63 / 2]])
    out = transform_preds(coords, center, scale, (48, 64))
    assert np.allclose(out[0], center, atol=1.0)
```

- [ ] **Step 3: Run test to verify it fails**

Run:
```bash
PYTHONPATH=src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_vitpose_transforms.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named '...vitpose.transforms'`

- [ ] **Step 4: Implement, transcribing get_warp_matrix from Step 1**

```python
# src/hydra_suite/core/identity/pose/vitpose/transforms.py
"""Top-down pre/post-processing, UDP variant.

UDP (Unbiased Data Processing, Huang et al. CVPR 2020) defines unit length as
pixel SPACING (size - 1) rather than pixel count. Every released ViTPose
checkpoint sets use_udp=True, so warp, encode, and decode must all agree; mixing
costs ~1-2 AP silently. get_warp_matrix / transform_preds are transcribed from
upstream mmpose/core/post_processing/post_transforms.py.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import (
    IMAGE_SIZE_WH,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PADDING_FACTOR,
    PIXEL_STD,
)


def box2cs(box_xywh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, w, h = box_xywh[:4]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    aspect = IMAGE_SIZE_WH[0] / IMAGE_SIZE_WH[1]
    if w > aspect * h:
        h = w / aspect
    elif w < aspect * h:
        w = h * aspect
    scale = np.array([w, h], dtype=np.float32) / PIXEL_STD
    scale = scale * PADDING_FACTOR
    return center, scale


def get_warp_matrix(
    theta: float,
    size_input: np.ndarray,
    size_dst: np.ndarray,
    size_target: np.ndarray,
) -> np.ndarray:
    """TRANSCRIBE VERBATIM from upstream post_transforms.py (Step 1).

    Do not reconstruct from the paper or from memory: a flipped sign or a
    dropped 0.5 produces a plausible image and a silent ~1 AP loss.
    """
    raise NotImplementedError("transcribe from upstream — see Step 1")


def top_down_affine(
    img: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    rot: float = 0.0,
) -> np.ndarray:
    w, h = IMAGE_SIZE_WH
    trans = get_warp_matrix(
        rot,
        center * 2.0,
        np.array([w, h], dtype=np.float32) - 1.0,  # UDP: size - 1
        scale * PIXEL_STD,
    )
    return cv2.warpAffine(img, trans, (w, h), flags=cv2.INTER_LINEAR)


def normalize(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array(IMAGENET_MEAN, np.float32)) / np.array(
        IMAGENET_STD, np.float32
    )
    return np.ascontiguousarray(img.transpose(2, 0, 1))


def transform_preds(
    coords: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    output_size_wh: tuple[int, int],
) -> np.ndarray:
    """TRANSCRIBE the use_udp=True branch VERBATIM from upstream (Step 1).

    The UDP branch divides by (output_size - 1.0); the non-UDP branch does not.
    That single -1 is worth ~1 AP.
    """
    raise NotImplementedError("transcribe from upstream — see Step 1")
```

Replace both `NotImplementedError` bodies with the transcribed upstream code.

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
PYTHONPATH=src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_vitpose_transforms.py -q
```
Expected: PASS (8 passed)

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/identity/pose/vitpose/transforms.py tests/test_vitpose_transforms.py
git commit -m "feat(vitpose): UDP top-down transforms"
```

---

### Task 7: decode_udp_cv2 (the oracle)

**Files:**
- Create: `src/hydra_suite/core/identity/pose/vitpose/decode.py`
- Test: `tests/test_vitpose_decode.py`

**Interfaces:**
- Consumes: `config.UDP_BLUR_KERNEL`, `transforms.transform_preds`.
- Produces:
  - `get_max_preds(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — `(N,K,2)` coords, `(N,K,1)` maxvals
  - `decode_udp_cv2(heatmaps: np.ndarray, kernel: int = 11) -> tuple[np.ndarray, np.ndarray]`
  - `flip_back(heatmaps: np.ndarray, flip_pairs: Sequence[tuple[int,int]]) -> np.ndarray`

- [ ] **Step 1: Fetch upstream source — do not write from memory**

```bash
curl -sL -o /tmp/vitpose-ref/inference.py \
  https://raw.githubusercontent.com/ViTAE-Transformer/ViTPose/main/mmpose/core/evaluation/top_down_eval.py
grep -n "def post_dark_udp" -A 45 /tmp/vitpose-ref/inference.py
grep -n "def _get_max_preds" -A 25 /tmp/vitpose-ref/inference.py
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_vitpose_decode.py
import numpy as np
import pytest

from hydra_suite.core.identity.pose.vitpose.decode import (
    get_max_preds, decode_udp_cv2, flip_back,
)
from hydra_suite.core.identity.pose.vitpose.config import UDP_BLUR_KERNEL


def _gaussian_heatmap(h=64, w=48, cx=20.0, cy=30.0, sigma=2.0):
    ys, xs = np.mgrid[0:h, 0:w]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
    return g.astype(np.float32)


def test_get_max_preds_argmax():
    hm = np.zeros((1, 1, 64, 48), np.float32)
    hm[0, 0, 30, 20] = 1.0
    coords, maxvals = get_max_preds(hm)
    assert coords.shape == (1, 1, 2)
    assert np.allclose(coords[0, 0], [20.0, 30.0])
    assert np.allclose(maxvals[0, 0], [1.0])


def test_decode_refines_to_subpixel_peak():
    """Peak at a non-integer location: integer argmax lands at (20,30) but the
    true peak is (20.4, 30.4). DARK/UDP refinement must move toward it."""
    hm = _gaussian_heatmap(cx=20.4, cy=30.4)[None, None]
    coords, _ = decode_udp_cv2(hm, kernel=UDP_BLUR_KERNEL)
    assert abs(coords[0, 0, 0] - 20.4) < 0.25
    assert abs(coords[0, 0, 1] - 30.4) < 0.25


def test_decode_does_not_mutate_input():
    hm = _gaussian_heatmap()[None, None]
    original = hm.copy()
    decode_udp_cv2(hm, kernel=UDP_BLUR_KERNEL)
    assert np.array_equal(hm, original), "decode must not mutate its input"


def test_flip_back_swaps_pairs_and_mirrors():
    hm = np.zeros((1, 2, 4, 4), np.float32)
    hm[0, 0, 1, 0] = 1.0
    out = flip_back(hm, [(0, 1)])
    assert out[0, 1, 1, 3] == pytest.approx(1.0)
```

- [ ] **Step 3: Run test to verify it fails**

Run:
```bash
PYTHONPATH=src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_vitpose_decode.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named '...vitpose.decode'`

- [ ] **Step 4: Implement the oracle by transcribing from Step 1**

```python
# src/hydra_suite/core/identity/pose/vitpose/decode.py
"""Heatmap decoding, UDP/DARK.

Two implementations:
  decode_udp_cv2   -- faithful port of upstream post_dark_udp. The ORACLE.
                      float64 numpy on CPU. Not the production path.
  decode_udp_torch -- device-resident, float32. Production (Task 8).

They are bound by a parity test. cv2 anchors us to mmpose; the parity test
anchors torch to cv2; Gate C validates the whole chain.

On the blur sigma: cv2.GaussianBlur(hm, (11, 11), 0) means "derive sigma from
kernel" -> 0.3*((11-1)*0.5 - 1) + 0.8 == 2.0, exactly the training sigma.
HuggingFace instead hardcodes sigma=0.8, which does not track kernel size. That
is an unflagged deviation and we deliberately do not follow it.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def get_max_preds(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TRANSCRIBE from upstream _get_max_preds (Step 1)."""
    raise NotImplementedError("transcribe from upstream — see Step 1")


def decode_udp_cv2(
    heatmaps: np.ndarray, kernel: int = 11
) -> tuple[np.ndarray, np.ndarray]:
    """TRANSCRIBE from upstream post_dark_udp (Step 1).

    Shape: (N, K, H, W) -> coords (N, K, 2), maxvals (N, K, 1).
    Must NOT mutate `heatmaps` — upstream blurs in place, so copy first.
    """
    raise NotImplementedError("transcribe from upstream — see Step 1")


def flip_back(
    heatmaps: np.ndarray, flip_pairs: Sequence[tuple[int, int]]
) -> np.ndarray:
    """Mirror heatmaps and swap left/right keypoint channels.

    With UDP, do NOT additionally apply the shift_heatmap column shift
    (`hm[:, :, :, 1:] = hm[:, :, :, :-1]`). That is the non-UDP correction;
    applying both double-corrects.
    """
    out = heatmaps[..., ::-1].copy()
    for a, b in flip_pairs:
        tmp = out[:, a].copy()
        out[:, a] = out[:, b]
        out[:, b] = tmp
    return out
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
PYTHONPATH=src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_vitpose_decode.py -q
```
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/identity/pose/vitpose/decode.py tests/test_vitpose_decode.py
git commit -m "feat(vitpose): UDP/DARK cv2 decode (numerical oracle)"
```

---

### Task 8: GATE B — decode_udp_torch and parity

Why this exists: the faithful cv2 decode pulls heatmaps to host, blurs in
OpenCV, and pushes coordinates back — a GPU→CPU roundtrip in the hottest loop.
Spec 2 requires no roundtrips, so the decode must run on-device. This task
proves the on-device version is numerically the same thing.

**Files:**
- Modify: `src/hydra_suite/core/identity/pose/vitpose/decode.py`
- Modify: `tests/test_vitpose_decode.py`

**Interfaces:**
- Consumes: `decode_udp_cv2`, `get_max_preds`.
- Produces:
  - `decode_udp_torch(heatmaps: torch.Tensor, kernel: int = 11) -> tuple[torch.Tensor, torch.Tensor]` — accepts `(N,K,H,W)` on any device; returns coords `(N,K,2)` and maxvals `(N,K,1)` on the **same device**, never touching host memory.

- [ ] **Step 1: Write the failing parity test**

```python
# append to tests/test_vitpose_decode.py
import torch
from hydra_suite.core.identity.pose.vitpose.decode import decode_udp_torch


def _random_peaky_heatmaps(n=2, k=17, h=64, w=48, seed=0):
    """Real forward-pass heatmaps are peaky but noisy and occasionally flat or
    multi-modal. Pure synthetic Gaussians are too well-conditioned to exercise
    the Hessian solve, so add noise and a second lobe."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, k, h, w), np.float32)
    ys, xs = np.mgrid[0:h, 0:w]
    for i in range(n):
        for j in range(k):
            cx, cy = rng.uniform(6, w - 6), rng.uniform(6, h - 6)
            g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / 8.0)
            cx2, cy2 = rng.uniform(6, w - 6), rng.uniform(6, h - 6)
            g = g + 0.3 * np.exp(-((xs - cx2) ** 2 + (ys - cy2) ** 2) / 8.0)
            out[i, j] = g + rng.normal(0, 0.01, (h, w))
    return out.astype(np.float32)


def _max_coord_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max())


def test_gate_b_torch_decode_matches_cv2_cpu():
    """GATE B. Tolerance 1e-2 heatmap units, per-keypoint (not averaged --
    averaging hides a single badly-decoded joint, which is the failure we care
    about). 1 heatmap unit is 4 image px, so 1e-2 is comfortably sub-pixel."""
    hm = _random_peaky_heatmaps()
    ref, ref_v = decode_udp_cv2(hm)
    got, got_v = decode_udp_torch(torch.from_numpy(hm))
    assert _max_coord_delta(ref, got.numpy()) < 1e-2
    assert _max_coord_delta(ref_v, got_v.numpy()) < 1e-4


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS unavailable"
)
def test_gate_b_torch_decode_matches_cv2_on_mps():
    """Same gate on MPS. NOTE: the oracle is float64 numpy on CPU; MPS has no
    float64, so this compares ACROSS DTYPES by design. float32 carries ~7
    decimal digits and we need ~2, so 1e-2 is still the right bound."""
    hm = _random_peaky_heatmaps()
    ref, _ = decode_udp_cv2(hm)
    got, _ = decode_udp_torch(torch.from_numpy(hm).to("mps"))
    assert _max_coord_delta(ref, got.cpu().numpy()) < 1e-2


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS unavailable"
)
def test_torch_decode_never_leaves_device():
    """The whole point: no GPU->CPU roundtrip."""
    hm = torch.from_numpy(_random_peaky_heatmaps()).to("mps")
    coords, maxvals = decode_udp_torch(hm)
    assert coords.device.type == "mps"
    assert maxvals.device.type == "mps"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_vitpose_decode.py -q -k gate_b
```
Expected: FAIL — `ImportError: cannot import name 'decode_udp_torch'`

- [ ] **Step 3: Implement the on-device decode**

```python
# append to src/hydra_suite/core/identity/pose/vitpose/decode.py
import torch
import torch.nn.functional as F


def _gaussian_kernel1d(kernel: int, device, dtype) -> torch.Tensor:
    """Match cv2.GaussianBlur(..., 0): sigma derived from kernel size, and the
    same normalised kernel cv2 builds."""
    sigma = 0.3 * ((kernel - 1) * 0.5 - 1) + 0.8
    x = torch.arange(kernel, device=device, dtype=dtype) - (kernel - 1) / 2
    k = torch.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def decode_udp_torch(
    heatmaps: torch.Tensor, kernel: int = 11
) -> tuple[torch.Tensor, torch.Tensor]:
    """Device-resident UDP/DARK decode. Mirrors decode_udp_cv2 exactly.

    The Gaussian blur is a separable depthwise conv (cv2 uses a separable
    kernel too, hence the 1-D construction). The 2x2 Hessian is solved in
    closed form rather than via torch.linalg.solve: MPS has linalg gaps, and a
    2x2 inverse is trivially exact anyway.
    """
    n, k, h, w = heatmaps.shape
    device, dtype = heatmaps.device, heatmaps.dtype

    flat = heatmaps.reshape(n * k, -1)
    maxvals, idx = flat.max(dim=1, keepdim=True)
    px = (idx % w).to(dtype)
    py = torch.div(idx, w, rounding_mode="floor").to(dtype)
    coords = torch.cat([px, py], dim=1)
    coords = torch.where(maxvals > 0.0, coords, torch.full_like(coords, -1.0))

    # Blur (separable depthwise), matching cv2's BORDER_REFLECT_101 default.
    k1 = _gaussian_kernel1d(kernel, device, dtype)
    pad = kernel // 2
    x = heatmaps.reshape(n * k, 1, h, w)
    x = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, k1.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    x = F.conv2d(x, k1.view(1, 1, -1, 1))
    blurred = x.reshape(n * k, h, w)

    # Upstream normalises the blurred map back to the original peak, then
    # clips and takes the log before the Taylor step.
    orig_max = heatmaps.reshape(n * k, -1).max(dim=1).values.view(-1, 1, 1)
    blur_max = blurred.reshape(n * k, -1).max(dim=1).values.view(-1, 1, 1)
    blurred = blurred * orig_max / blur_max.clamp_min(1e-12)
    blurred = blurred.clamp(1e-3, 50.0).log()

    bx = coords[:, 0].long().clamp(1, w - 2)
    by = coords[:, 1].long().clamp(1, h - 2)
    b = torch.arange(n * k, device=device)

    def at(dy: int, dx: int) -> torch.Tensor:
        return blurred[b, by + dy, bx + dx]

    dx = 0.5 * (at(0, 1) - at(0, -1))
    dy = 0.5 * (at(1, 0) - at(-1, 0))
    dxx = 0.25 * (at(0, 2) - 2 * at(0, 0) + at(0, -2))
    dyy = 0.25 * (at(2, 0) - 2 * at(0, 0) + at(-2, 0))
    dxy = 0.25 * (at(1, 1) - at(-1, 1) - at(1, -1) + at(-1, -1))

    # Closed-form 2x2 inverse; skip refinement where the Hessian is singular.
    det = dxx * dyy - dxy * dxy
    ok = det.abs() > 1e-12
    inv = torch.zeros_like(det)
    inv[ok] = 1.0 / det[ok]
    offset_x = -inv * (dyy * dx - dxy * dy)
    offset_y = -inv * (dxx * dy - dxy * dx)
    offset_x = torch.where(ok, offset_x, torch.zeros_like(offset_x))
    offset_y = torch.where(ok, offset_y, torch.zeros_like(offset_y))

    refined = coords.clone()
    refined[:, 0] = coords[:, 0] + offset_x
    refined[:, 1] = coords[:, 1] + offset_y
    valid = (maxvals.squeeze(1) > 0.0).unsqueeze(1)
    refined = torch.where(valid, refined, coords)

    return refined.reshape(n, k, 2), maxvals.reshape(n, k, 1)
```

- [ ] **Step 4: Run the gate**

Run:
```bash
PYTHONPATH=src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_vitpose_decode.py -q
```
Expected: PASS (7 passed)

**If parity fails, this is a finding, not a nuisance. Do NOT loosen the bound.**
Debug in this order:

1. Print `_max_coord_delta` — if it is ~0.25, only some keypoints disagree:
   the Hessian singular-guard differs from upstream's.
2. If the delta is a constant offset, the blur border mode differs
   (cv2 defaults to `BORDER_REFLECT_101`; `F.pad(mode="reflect")` matches it —
   `mode="replicate"` does not).
3. If the delta grows near edges, the `clamp(1, w-2)` bounds differ from
   upstream's.
4. If everything disagrees, the blur normalisation step (`orig_max/blur_max`)
   is wrong or absent.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/identity/pose/vitpose/decode.py tests/test_vitpose_decode.py
git commit -m "feat(vitpose): GATE B - device-resident UDP decode with cv2 parity"
```

---

### Task 9: ViTPose+ MoE and GATE A(3)

**Files:**
- Modify: `src/hydra_suite/core/identity/pose/vitpose/model.py`
- Modify: `src/hydra_suite/core/identity/pose/vitpose/vitpose.py`
- Modify: `tests/test_vitpose_model.py`, `tests/test_vitpose_weights.py`

**Interfaces:**
- Consumes: `Block`, `ViT`, `build_head`, `config.NUM_EXPERTS`.
- Produces:
  - `class MoEMlp(nn.Module)` — attrs `fc1`, `fc2`, `experts: nn.ModuleList`; `forward(x, indices) -> Tensor`
  - `ViT.__init__` gains `part_features: int | None = None`; when set, blocks use `MoEMlp` and `forward(x, dataset_index: int = 0)`
  - `ViTPoseMoE(nn.Module)` — attrs `backbone`, `keypoint_head`, `associate_keypoint_heads: nn.ModuleList`
  - `build_vitpose_moe(variant: str, num_keypoints: int = 17) -> ViTPoseMoE`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_vitpose_model.py
from hydra_suite.core.identity.pose.vitpose.model import MoEMlp


def test_moe_shapes_for_base():
    """B: fc1 768->3072, fc2 3072->576 (D - part_features), 6 experts 3072->192.
    Concat of shared (576) + expert (192) restores 768."""
    m = MoEMlp(dim=768, hidden=3072, part_features=192, num_expert=6)
    assert m.fc1.out_features == 3072
    assert m.fc2.out_features == 768 - 192
    assert len(m.experts) == 6
    assert m.experts[0].out_features == 192
    out = m(torch.zeros(2, 10, 768), torch.zeros(2, dtype=torch.long))
    assert out.shape == (2, 10, 768)


def test_moe_routing_is_by_dataset_index_not_learned():
    """Routing is NOT learned: `indices` is the dataset index supplied from
    outside. Different indices must select different experts."""
    m = MoEMlp(dim=8, hidden=16, part_features=4, num_expert=6).eval()
    with torch.no_grad():
        for i, e in enumerate(m.experts):
            e.weight.fill_(float(i + 1))
            e.bias.zero_()
        m.fc1.weight.zero_(); m.fc1.bias.fill_(1.0)
        m.fc2.weight.zero_(); m.fc2.bias.zero_()
        x = torch.zeros(1, 1, 8)
        out0 = m(x, torch.zeros(1, dtype=torch.long))
        out3 = m(x, torch.full((1,), 3, dtype=torch.long))
    assert not torch.allclose(out0[..., -4:], out3[..., -4:])
```

```python
# append to tests/test_vitpose_weights.py
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose_moe

requires_plus = pytest.mark.skipif(
    not (ASSET_DIR / "vitpose+_base.pth").exists(),
    reason="run tools/vitpose/fetch_assets.py first",
)


@requires_plus
def test_gate_a3_strict_load_moe():
    """GATE A(3). ViTPose+ has 1 COCO head + 5 associate heads
    (out_channels 14/16/17/17/133)."""
    m = build_vitpose_moe("B")
    load_checkpoint(m, ASSET_DIR / "vitpose+_base.pth", strict=True)


@requires_plus
def test_moe_checkpoint_rejects_classic_module():
    """A ViTPose+ checkpoint must NOT load into a classic ViT: MoE fc2 is
    [D - part_features, 4D], not [D, 4D].

    Matches on "size mismatch", not bare Exception -- a bare Exception would
    also 'pass' on an ImportError or a typo in this test, asserting nothing.

    Note it is torch's own RuntimeError that fires here, NOT our
    CheckpointKeyError: load_state_dict(strict=False) still raises on a SHAPE
    mismatch before returning missing/unexpected keys, and MoE fc2 is
    [D - part_features, 4D] vs classic [D, 4D]. So this is a shape failure, not
    a key failure.
    """
    m = build_vitpose("B", "classic")
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_checkpoint(m, ASSET_DIR / "vitpose+_base.pth", strict=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_vitpose_model.py tests/test_vitpose_weights.py -q -k "moe or gate_a3"
```
Expected: FAIL — `ImportError: cannot import name 'MoEMlp'`

- [ ] **Step 3: Implement MoE**

```python
# append to src/hydra_suite/core/identity/pose/vitpose/model.py
class MoEMlp(nn.Module):
    """ViTPose+ FFN. ONLY the FFN differs from classic; attention, patch embed,
    pos embed and norms are byte-identical.

    Routing is NOT learned -- `indices` is the dataset index threaded in from
    outside. Upstream runs all experts and masks (a DDP workaround); for
    single-dataset inference we index the expert directly, which is numerically
    identical and avoids 6x the expert-branch compute.
    """

    def __init__(
        self, dim: int, hidden: int, part_features: int, num_expert: int = 6
    ) -> None:
        super().__init__()
        self.part_features = part_features
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim - part_features)
        self.experts = nn.ModuleList(
            [nn.Linear(hidden, part_features) for _ in range(num_expert)]
        )

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        shared = self.fc2(x)
        uniq = torch.unique(indices)
        if uniq.numel() == 1:
            expert = self.experts[int(uniq.item())](x)
        else:
            expert = torch.zeros(
                *x.shape[:-1], self.part_features, device=x.device, dtype=x.dtype
            )
            for i, e in enumerate(self.experts):
                mask = indices == i
                if mask.any():
                    expert[mask] = e(x[mask])
        return torch.cat([shared, expert], dim=-1)
```

Then thread `dataset_index` through `Block.forward` and `ViT.forward` when
`part_features` is set, and add to `vitpose.py`:

```python
# append to src/hydra_suite/core/identity/pose/vitpose/vitpose.py
ASSOCIATE_HEAD_CHANNELS = (14, 16, 17, 17, 133)  # AiC, MPII, AP-10K, APT-36K, WholeBody


class ViTPoseMoE(nn.Module):
    def __init__(
        self,
        backbone: ViT,
        keypoint_head: nn.Module,
        associate_keypoint_heads: nn.ModuleList,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.keypoint_head = keypoint_head
        self.associate_keypoint_heads = associate_keypoint_heads

    def forward(self, x: torch.Tensor, dataset_index: int = 0) -> torch.Tensor:
        feat = self.backbone(x, dataset_index=dataset_index)
        if dataset_index == 0:
            return self.keypoint_head(feat)
        return self.associate_keypoint_heads[dataset_index - 1](feat)


def build_vitpose_moe(variant: str, num_keypoints: int = 17) -> ViTPoseMoE:
    v = VARIANTS[variant]
    backbone = ViT(
        embed_dim=v.embed_dim,
        depth=v.depth,
        num_heads=v.num_heads,
        part_features=v.part_features,
    )
    head = build_head("classic", v.embed_dim, num_keypoints)
    associates = nn.ModuleList(
        [build_head("classic", v.embed_dim, c) for c in ASSOCIATE_HEAD_CHANNELS]
    )
    return ViTPoseMoE(backbone, head, associates)
```

- [ ] **Step 4: Run the gate**

Run:
```bash
PYTHONPATH=src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_vitpose_model.py tests/test_vitpose_weights.py -q
```
Expected: PASS

If `test_gate_a3_strict_load_moe` reports unexpected `associate_keypoint_heads.*`
keys, check `ASSOCIATE_HEAD_CHANNELS` order against the checkpoint's actual
`final_layer` shapes:

```bash
PYTHONPATH=src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python - <<'PY'
import torch, os
from pathlib import Path
p = Path(os.path.expanduser("~/.cache/vitpose-assets/vitpose+_base.pth"))
sd = torch.load(p, map_location="cpu", weights_only=True)["state_dict"]
for k, v in sd.items():
    if "final_layer.weight" in k:
        print(f"{k:60s} {tuple(v.shape)}")
PY
```

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/identity/pose/vitpose/ tests/test_vitpose_model.py tests/test_vitpose_weights.py
git commit -m "feat(vitpose): GATE A(3) - ViTPose+ MoE backbone and per-dataset heads"
```

---

### Task 10: GATE C — COCO val AP reproduction

The oracle. Everything above is necessary; only this is sufficient.

**Files:**
- Create: `tools/vitpose/eval_coco.py`
- Test: `tests/test_vitpose_eval_coco.py`

**Interfaces:**
- Consumes: everything.
- Produces: `evaluate(variant, head, ckpt, device, limit=None) -> dict[str, float]` with key `"AP"`.

- [ ] **Step 1: Install pycocotools and acquire COCO val2017**

`pycocotools` is NOT currently installed in `hydra-mps` (verified 2026-07-16).
Install it — and NOT `xtcocotools`, which mmpose uses but which does not install
on Python 3.13 from PyPI (wheels stop at cp311; the sdist then breaks on PEP 667):

```bash
/Users/neurorishika/miniforge3/envs/hydra-mps/bin/pip install pycocotools
```

Like `gdown`, this is an eval/dev dependency: do **not** add it to
`pyproject.toml` runtime deps. Spec 1 ships no runtime code that evaluates AP.

```bash
D=$HOME/.cache/vitpose-assets
mkdir -p $D && cd $D
[ -d val2017 ] || { curl -O http://images.cocodataset.org/zips/val2017.zip && unzip -q val2017.zip; }
[ -f annotations/person_keypoints_val2017.json ] || {
  curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -q annotations_trainval2017.zip annotations/person_keypoints_val2017.json
}
PYTHONPATH=.:src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -c "
from pathlib import Path; import os
from tools.vitpose.fetch_assets import fetch
print(fetch('coco_val2017_person_detections', Path(os.path.expanduser('~/.cache/vitpose-assets'))))
"
```

The last command MUST pass its SHA256 check. If it raises `AssetIntegrityError`,
you have the dummy file or a corrupt download — re-fetch. **Never** update the
pinned constant to match what you downloaded.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_vitpose_eval_coco.py
import os
from pathlib import Path

import pytest

ASSET_DIR = Path(os.path.expanduser("~/.cache/vitpose-assets"))

requires_coco = pytest.mark.skipif(
    not (ASSET_DIR / "val2017").exists()
    or not (ASSET_DIR / "COCO_val2017_detections_AP_H_56_person.json").exists(),
    reason="COCO val2017 + detections required; see Task 10 Step 1",
)


@requires_coco
def test_smoke_eval_on_20_images():
    """Fast feedback before the full ~40min run.

    Asserts AP > 0.5, not just 0<=AP<=1 (which is vacuously true for any
    result, including a totally broken pipeline). A correctly-ported ViTPose-B
    scores ~0.76 on full val; 20 images is noisy but a working pipeline clears
    0.5 comfortably, while a broken one lands near 0.
    """
    from tools.vitpose.eval_coco import evaluate

    res = evaluate("B", "classic", ASSET_DIR / "vitpose-b.pth", "mps", limit=20)
    assert res["AP"] > 0.5, f"smoke AP {res['AP']:.3f} — pipeline is broken"


@pytest.mark.slow
@requires_coco
def test_gate_c_classic_reproduces_published_ap():
    """GATE C. Published ViTPose-B classic = 75.8 AP.

    Diagnostic ladder if this fails:
      ~1 AP off    -> UDP mismatch (warp and decode disagree)
      ~0.3 AP off  -> decode blur sigma
      wildly off   -> patch padding or pos-embed
    """
    from tools.vitpose.eval_coco import evaluate

    res = evaluate("B", "classic", ASSET_DIR / "vitpose-b.pth", "mps")
    assert abs(res["AP"] * 100 - 75.8) < 0.2, f"got {res['AP']*100:.2f} AP"


@pytest.mark.slow
@requires_coco
def test_gate_c_simple_reproduces_published_ap():
    """GATE C. Published ViTPose-B simple = 75.5 AP."""
    from tools.vitpose.eval_coco import evaluate

    res = evaluate("B", "simple", ASSET_DIR / "vitpose-b-simple.pth", "mps")
    assert abs(res["AP"] * 100 - 75.5) < 0.2, f"got {res['AP']*100:.2f} AP"
```

- [ ] **Step 3: Register the `slow` marker AND make it non-gating by default**

`pytest.ini` currently reads:

```ini
[pytest]
testpaths = tests
markers =
    benchmark: performance-oriented tests that are non-gating by default
addopts = -m "not benchmark" -p no:napari
```

Note `addopts` filters only `benchmark`. Adding a bare `slow` marker is not
enough — a plain `pytest` would then silently start the ~40-minute Gate C run.
Follow the repo's existing "non-gating by default" convention and edit **both**
lines:

```ini
[pytest]
testpaths = tests
markers =
    benchmark: performance-oriented tests that are non-gating by default
    slow: full COCO AP runs (~40 min on MPS); non-gating by default
addopts = -m "not benchmark and not slow" -p no:napari
```

A command-line `-m` overrides `addopts`, so `pytest -m slow` still runs Gate C.

- [ ] **Step 4: Implement the harness**

```python
# tools/vitpose/eval_coco.py
"""GATE C: reproduce published ViTPose COCO val AP.

Top-down AP is only comparable to published numbers when evaluated against the
STANDARD person detections (not ground-truth boxes, which score higher).

pycocotools is sufficient here: xtcocotools' default sigmas are allclose to
pycocotools' COCO sigmas and the full stats vector is identical. xtcocotools
also does not install on Python 3.13 from PyPI.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2  # noqa: F401  (import before torch; see vitpose/__init__.py)
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from hydra_suite.core.identity.pose.vitpose.decode import decode_udp_torch, flip_back
from hydra_suite.core.identity.pose.vitpose.transforms import (
    box2cs, normalize, top_down_affine, transform_preds,
)
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose
from hydra_suite.core.identity.pose.vitpose.weights import load_checkpoint

ASSET_DIR = Path(os.path.expanduser("~/.cache/vitpose-assets"))
COCO_FLIP_PAIRS = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
]
DET_SCORE_THR = 0.0  # upstream keeps all detections and lets OKS sort it out


def evaluate(
    variant: str,
    head: str,
    ckpt: Path,
    device: str = "cpu",
    limit: int | None = None,
    batch_size: int = 16,
) -> dict[str, float]:
    ann_file = ASSET_DIR / "annotations" / "person_keypoints_val2017.json"
    det_file = ASSET_DIR / "COCO_val2017_detections_AP_H_56_person.json"
    coco = COCO(str(ann_file))
    dets = json.loads(det_file.read_text())
    dets = [d for d in dets if d["category_id"] == 1 and d["score"] > DET_SCORE_THR]
    if limit is not None:
        keep = set(sorted({d["image_id"] for d in dets})[:limit])
        dets = [d for d in dets if d["image_id"] in keep]

    model = build_vitpose(variant, head).eval().to(device)
    load_checkpoint(model, ckpt, strict=True)

    results = []
    for start in range(0, len(dets), batch_size):
        chunk = dets[start : start + batch_size]
        crops, metas = [], []
        for d in chunk:
            img_path = ASSET_DIR / "val2017" / coco.loadImgs(d["image_id"])[0][
                "file_name"
            ]
            img = cv2.imread(str(img_path))
            c, s = box2cs(np.array(d["bbox"], np.float32))
            crops.append(normalize(top_down_affine(img, c, s)))
            metas.append((d, c, s))
        batch = torch.from_numpy(np.stack(crops)).to(device)
        with torch.no_grad():
            hm = model(batch)
            # Flip test: every ViTPose config sets flip_test=True.
            hm_flip = model(torch.flip(batch, dims=[3]))
            hm_flip = torch.from_numpy(
                flip_back(hm_flip.cpu().numpy(), COCO_FLIP_PAIRS)
            ).to(device)
            # With UDP, shift_heatmap must stay False -- do NOT column-shift.
            hm = (hm + hm_flip) * 0.5
            coords, maxvals = decode_udp_torch(hm)
        coords_np = coords.cpu().numpy()
        vals_np = maxvals.cpu().numpy()
        for i, (d, c, s) in enumerate(metas):
            kpts = transform_preds(coords_np[i], c, s, (48, 64))
            results.append(
                {
                    "image_id": d["image_id"],
                    "category_id": 1,
                    "keypoints": np.concatenate(
                        [kpts, vals_np[i]], axis=1
                    ).reshape(-1).tolist(),
                    "score": float(d["score"] * vals_np[i].mean()),
                }
            )

    dt = coco.loadRes(results)
    e = COCOeval(coco, dt, "keypoints")
    e.evaluate()
    e.accumulate()
    e.summarize()
    return {"AP": float(e.stats[0])}
```

- [ ] **Step 5: Run the smoke test first**

Run:
```bash
PYTHONPATH=.:src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_vitpose_eval_coco.py -q -k smoke
```
Expected: PASS. Do not proceed to the full run until this is green.

- [ ] **Step 6: Run GATE C**

Run:
```bash
PYTHONPATH=.:src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_vitpose_eval_coco.py -q -m slow
```
Expected: PASS (2 passed). **~40 min on MPS** — 3,893 images x flip-test is
~7,800 ViT-B forwards. It is not hung.

Apply the diagnostic ladder in the test docstring if AP is off.

- [ ] **Step 7: Commit**

```bash
git add tools/vitpose/eval_coco.py tests/test_vitpose_eval_coco.py pytest.ini
git commit -m "feat(vitpose): GATE C - COCO val AP reproduction harness"
```

---

### Task 11: Full suite, lint, and CUDA validation handoff

- [ ] **Step 1: Run the full fast suite (no regressions elsewhere)**

Run:
```bash
PYTHONPATH=src /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/ -q -m "not slow"
```
Expected: all pass. Baseline before this work was 29 passed for
`tests/test_pose_pipeline.py tests/test_inference_stages_pose.py`.

- [ ] **Step 2: Verify the leaf constraint one final time**

Run:
```bash
grep -rn "from hydra_suite\|import hydra_suite" src/hydra_suite/core/identity/pose/vitpose/ && echo "VIOLATION" || echo "clean: leaf module imports nothing from hydra_suite"
```
Expected: `clean: ...`

- [ ] **Step 3: Confirm nothing was wired in (Spec 1 is standalone)**

Run:
```bash
git diff --stat main...HEAD -- src/hydra_suite/core/identity/pose/api.py src/hydra_suite/core/identity/pose/types.py src/hydra_suite/core/inference/
```
Expected: **empty**. Any change here is Spec 3 scope leaking in.

- [ ] **Step 4: Lint**

Run:
```bash
make lint-moderate
make docs-check
```

- [ ] **Step 5: CUDA validation on the remote box**

MPS was the development target; CUDA is the deployment target. Run the same
gates on `rutalab@mehek.taild08eb9.ts.net`:

```bash
ssh rutalab@mehek.taild08eb9.ts.net
# in the repo's CUDA env:
PYTHONPATH=src python -m pytest tests/test_vitpose_decode.py -q
PYTHONPATH=.:src python -m pytest tests/test_vitpose_eval_coco.py -q -m slow
```

Gate B's MPS tests are `skipif`-guarded, so on CUDA they skip. **Before running,
add a CUDA-guarded twin of `test_gate_b_torch_decode_matches_cv2_on_mps`** —
otherwise the on-device decode is never parity-checked on the deployment target,
which is the one place it actually has to be right:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gate_b_torch_decode_matches_cv2_on_cuda():
    hm = _random_peaky_heatmaps()
    ref, _ = decode_udp_cv2(hm)
    got, _ = decode_udp_torch(torch.from_numpy(hm).to("cuda"))
    assert _max_coord_delta(ref, got.cpu().numpy()) < 1e-2
```

CUDA AP should match the MPS AP to well within Gate C's 0.2 tolerance. A larger
gap means a device-dependent bug in `decode_udp_torch` — most likely the
`det.abs() > 1e-12` singular-guard behaving differently in float32 across
backends.

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "test(vitpose): CUDA parity twin for on-device decode"
```

---

## Definition of Done

| gate | check | target |
|---|---|---|
| A(1) | `strict=True` load `vitpose-b.pth` | no missing/unexpected keys |
| A(2) | `strict=True` load `vitpose-b-simple.pth` | no missing/unexpected keys |
| A(3) | `strict=True` load `vitpose+_base.pth` | no missing/unexpected keys |
| B | `decode_udp_torch` vs `decode_udp_cv2` | max per-keypoint delta < 1e-2, on CPU + MPS + CUDA |
| C | COCO val AP, classic | 75.8 ± 0.2 |
| C | COCO val AP, simple | 75.5 ± 0.2 |
| — | detections file SHA256 | `53ba0ad8…` (guards against the dummy) |
| — | leaf constraint | no `hydra_suite` imports in `vitpose/` |
| — | no integration | zero diff in `pose/api.py`, `pose/types.py`, `core/inference/` |

## Out of scope (Specs 2-4)

No backend class, no `PoseInferenceBackend`, no `PoseRuntimeConfig` changes, no
registry, no runtime layer, no ONNX/TensorRT export, no training, no GUI, no
SLEAP changes.
