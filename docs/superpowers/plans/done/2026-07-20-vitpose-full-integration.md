# ViTPose Full Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ViTPose a selectable pose backend that runs in the live tracking pipeline on all four runtimes (torch cpu/mps/cuda, TensorRT, CoreML), so a user can fine-tune a ViTPose model and immediately track with it.

**Architecture:** A thin `backends/vitpose.py` (mirroring the 294-line `yolo.py`, not the 1743-line `sleap.py`) drives the Spec-1 leaf (`core/identity/pose/vitpose/`): compose numpy/cv2 preprocess → torch/engine forward → on-device `decode_udp_torch` → numpy inverse. Runtime selection flows through Gen-2 (`runtime_tier` → `ResolvedBackend`), translated to the pose layer's `(device, runtime_flavor)` strings at `stages/pose.py` exactly as SLEAP/YOLO already are. Accelerated runtimes use a shared `core/identity/pose/runtime/` package *moved out of* `sleap.py` (not duplicated) plus a new CoreML runner.

**Tech Stack:** PyTorch (fp32), onnxruntime, tensorrt, coremltools; the Spec-1 ViTPose leaf; Gen-2 runtime resolver.

**Sequencing note (deliberate deviation from the design spec's Phase A-first order):** This plan delivers the **native path + full family integration first** (Phase 1 → train→track works on cpu/gpu/mps), then the **accelerated runtimes** (Phase 2), then **parity + e2e** (Phase 3). The design spec's own "Dependencies & sequencing" section blesses native-first; it front-loads a usable ViTPose and defers the one risky change (refactoring live SLEAP) until ViTPose is already provable.

## Global Constraints

Every task's requirements implicitly include these. Exact values, copied from the design spec and repo conventions:

- **Environment:** use the `hydra-mps` conda env (base env torch is broken). Do **not** run `make format` (broken: black `pathspec.patterns.gitignore` error); run `black <files>` and `isort <files>` directly on the task's files.
- **Leaf purity:** everything under `core/identity/pose/vitpose/` imports nothing from `hydra_suite` app layers; the new `core/identity/pose/runtime/` and `backends/vitpose.py` live in Core and must not import from any app layer (posekit/trackerkit) — CLAUDE.md dependency direction.
- **Precision: FP32 everywhere.** No fp16. (`build_tensorrt_engine(fp16=False)` default; keypoint precision.) Matches every leaf export default.
- **Fixed geometry:** `IMAGE_SIZE_WH = (192, 256)` (W,H), `HEATMAP_SIZE_WH = (48, 64)`, from `vitpose/config.py`. ViTPose input is `(B, 3, 256, 192)`; heatmaps `(B, K, 64, 48)`.
- **Runtime vocabulary:** the pose layer's `runtime_flavor` for ViTPose is one of `native | tensorrt | coreml` (there is no `onnx` tier post-Gen-2; ONNX Runtime is only the TRT-EP/CoreML-EP fallback mechanism). `ResolvedBackend.backend ∈ {torch, tensorrt, coreml}`, `device ∈ {cpu, cuda, mps}`.
- **No MoE / ViTPose+**, no new runtime tiers, no entry-point plugin discovery. Registry is a plain dict.
- **Artifact convention:** co-locate the exported artifact with the source checkpoint; write the `<artifact>.runtime_meta.json` sidecar only after a successful export; the signature includes a **recipe-version tag** (`vitpose-v1`).
- **Paths:** use `hydra_suite.paths` (`get_models_dir`, etc.); never `Path(__file__).parents[N]`.
- **Pre-commit per task:** run the task's focused tests, then `black`/`isort` on changed files. Pre-PR (final): `make commit-prep`, `make lint-moderate`, `make docs-check`.
- **Line length 88.** Follow `yolo.py` as the size/style model.

## Verified existing APIs (consume these verbatim — do not guess)

**Gen-2 runtime** (`hydra_suite.runtime`):
- `runtime/resolver.py`: `ResolvedBackend(backend: Literal["torch","tensorrt","coreml"], device: Literal["cpu","cuda","mps"], used_fallback: bool)` (`:24`); `STAGES = ("obb","head_tail","cnn","yolo_pose","sleap_pose","bgsub")` (`:15`); `RuntimeResolver(tier, platform).resolve(stage, artifact_available=lambda: True)` (`:44`).
- `runtime/onnx_providers.py`: `execution_providers_for(resolved, include_cpu_fallback: bool = True) -> List[object]` (`:60`).
- `core/inference/runtime.py`: `RuntimeContext.resolved: ResolvedBackend | None` (`:59`); `resolved_backend_for(runtime: RuntimeContext) -> ResolvedBackend` (`:158`).

**Pose leaf** (`core/identity/pose/vitpose/`):
- `vitpose.py:27` `build_vitpose(variant: str, head: str, num_keypoints: int = 17) -> ViTPose`; `ViTPose.forward(x: torch.Tensor) -> torch.Tensor` (heatmaps `(n,k,64,48)`).
- `weights.py:11` `class CheckpointKeyError(RuntimeError)`; `weights.py:15` `load_checkpoint(model, path: Path, strict: bool = True) -> None` (reads `blob["state_dict"]` else `blob`).
- `export.py:24` `class ExportError(RuntimeError)`; `export_onnx(model, path, *, opset=17, dynamic_batch=True, dataset_index=None) -> Path` (`:49`); `build_tensorrt_engine(onnx_path, engine_path, *, fp16=False, workspace_gb=4.0, max_batch=64) -> Path` (`:129`); `export_coreml(model, path, *, compute_units="ALL") -> Path` (`:219`).
- `decode.py:185` `decode_udp_torch(heatmaps: torch.Tensor, kernel=11) -> tuple[torch.Tensor, torch.Tensor]` (coords `(n,k,2)`, maxvals `(n,k,1)`; device-resident). `decode.py:137` `decode_udp_cv2(heatmaps: np.ndarray, kernel=11) -> tuple[np.ndarray, np.ndarray]` (oracle).
- `transforms.py`: `box2cs(box_xywh) -> (center (2,), scale (2,))` (`:26`); `top_down_affine(img, center, scale, rot=0.0) -> np.ndarray` (`:96`); `normalize(img_bgr) -> (3,H,W) float32` (`:107`); `transform_preds(coords, center, scale, output_size_wh) -> (K,ndims)` (`:115`).
- `config.py`: `IMAGE_SIZE_WH=(192,256)`, `HEATMAP_SIZE_WH=(48,64)`, `PADDING_FACTOR=1.25`; `VARIANTS: dict[str, ViTPoseVariant]` keyed `"S"/"B"/"L"/"H"` with fields `embed_dim, depth, num_heads, part_features, drop_path_rate, layer_decay`.
- Training checkpoint (`training/train.py:131`): `{"model_state", "optim_state", "variant", "num_keypoints", "epoch", "pck", "sched_state"}`. Always classic head (`training/model_setup.py:19` hardcodes `"classic"`).

**Pose family** (`core/identity/pose/`):
- `types.py:56` `PoseInferenceBackend` Protocol: attr `output_keypoint_names: List[str]`; `preferred_input_size` property `-> int`; `warmup() -> None`; `predict_batch(crops: Sequence[np.ndarray]) -> List[PoseResult]`; `close() -> None`.
- `types.py:11` `PoseResult(keypoints: Optional[np.ndarray], mean_conf: float, valid_fraction: float, num_valid: int, num_keypoints: int)`. Helper `summarize_keypoints(...)` and `empty_pose_result(...)` exist in the pose package (used by yolo/sleap; grep `pose/utils.py`).
- `types.py:22` `PoseRuntimeConfig` (flat): `backend_family, runtime_flavor="auto", device="auto", batch_size=4, model_path="", exported_model_path="", out_root=".", min_valid_conf=0.2, yolo_*, sleap_*, keypoint_names, skeleton_edges`.
- `api.py:32` `create_pose_backend_from_config(config: PoseRuntimeConfig) -> PoseInferenceBackend`; string dispatch on `config.backend_family`; raises `RuntimeError(f"Unsupported pose backend family: {backend_family}")` at `:159`.
- `artifacts.py`: `path_fingerprint_token(path_str: str) -> str` (`:71`); `artifact_meta_path(path: Path) -> Path` (`:82`); `artifact_meta_matches(path: Path, signature: str) -> bool` (`:92`); `write_artifact_meta(path: Path, signature: str) -> None` (`:107`); sidecar content `{"signature": str}`.

**Inference config** (`core/inference/`):
- `config.py:178` `PoseYOLOConfig`; `:187` `PoseSLEAPConfig`; `:195` `PoseConfig(backend: Literal["yolo","sleap"]="yolo", skeleton_file, yolo, sleap, crop_padding=0.1, ...)`; `_dict_to_config` pops `yolo`/`sleap` sub-dicts (`:346-358`). `InferenceConfig.runtime_tier: RuntimeTier = "gpu"` (`:241`).
- `cache/keys.py:166` `pose_cache_key(config: PoseConfig) -> CacheKey` branches `config.backend == "yolo"` else sleap.

**Stage bridge** (`core/inference/stages/pose.py`): `resolved = resolved_backend_for(runtime)` (`:98`); SLEAP translation branch (`:141-182`) maps `resolved` → `(device, runtime_flavor)` strings → `PoseRuntimeConfig` → `create_pose_backend_from_config`; YOLO branch (`:106-120`). Dispatch `on_cuda = batch.crops.is_cuda and hasattr(model.backend, "predict_batch_cuda")` (`:327`); dispatch (`:385-390`). `run_pose` non-batch path always `predict_batch` (`:248`).

**Backends to mirror:**
- `backends/yolo.py:38` `auto_export_yolo_model(config: PoseRuntimeConfig, runtime_flavor: str, runtime_device: Optional[str] = None) -> str`; `YoloNativeBackend.__init__(model_path, device="auto", min_valid_conf=0.2, keypoint_names=None, conf=1e-4, iou=0.7, max_det=1, batch_size=4)` (`:124`).
- `backends/sleap.py`: `_DirectOnnxSession(model_path: Path, resolved: ResolvedBackend)` (`:368`, providers via `execution_providers_for`); `_DirectTensorRTEngine(model_path: Path)` (`:513`, `run`/`run_cuda`); `_init_tensorrt_runner(model_path) -> engine|session` (`:838`, branches on suffix); `_ort_trt_ep_fallback()` (`:923`); `_build_trt_engine_from_onnx(onnx_path, engine_path, workspace_gb=4.0, fixed_hw=None) -> bool` (`:399`); `auto_export_sleap_model(config, runtime_flavor) -> str` (`:1407`).

## File Structure

**New files:**
- `src/hydra_suite/core/identity/pose/vitpose/adapter.py` — `load_finetuned_checkpoint` (checkpoint → module + metadata, head inference). *In the leaf; PoseKit-free.*
- `src/hydra_suite/core/identity/pose/vitpose/infer.py` — the pure crop→keypoints driver (compose pre/forward/decode/inverse + batching). *In the leaf.*
- `src/hydra_suite/core/identity/pose/backends/vitpose.py` — `ViTPoseBackend` (Protocol impl) + `auto_export_vitpose_model`.
- `src/hydra_suite/core/identity/pose/runtime/__init__.py`
- `src/hydra_suite/core/identity/pose/runtime/onnx_session.py` — `OnnxSessionRunner` (moved from sleap).
- `src/hydra_suite/core/identity/pose/runtime/tensorrt_engine.py` — `TensorRTEngineRunner` + `build_trt_engine_from_onnx` (moved).
- `src/hydra_suite/core/identity/pose/runtime/coreml_runner.py` — `CoreMLRunner` (new).
- `src/hydra_suite/core/identity/pose/runtime/accelerated.py` — `build_accelerated_runner` fallback ladder (moved).
- `tools/equivalence/verify_vitpose_runtimes.py` — parity harness.
- Test files under `tests/` per task.

**Modified files:**
- `vitpose/config.py` (training) — no change (head inferred, not stored).
- `pose/types.py` — add `vitpose_*` fields to `PoseRuntimeConfig`.
- `pose/api.py` — add `vitpose` factory branch.
- `pose/backends/sleap.py` — repoint runtime imports to `pose/runtime/` (Phase 2).
- `core/inference/config.py` — add `PoseViTPoseConfig`, extend `PoseConfig.backend` Literal, pop `vitpose` sub-dict.
- `core/inference/cache/keys.py` — add `vitpose` branch.
- `runtime/resolver.py` — add `"vitpose_pose"` to `STAGES`.
- `core/inference/stages/pose.py` — add `vitpose` translation branch.
- `posekit/gui/main_window.py`, `trackerkit/gui/orchestrators/config.py`, `trackerkit/gui/panels/detection_panel.py` — family pickers.

---

# Phase 1 — Native ViTPose backend + full family integration

Delivers: select ViTPose in the GUI, load a fine-tuned `best.pt`, track a video on cpu/gpu/mps.

## Task 1: Fine-tuned checkpoint adapter (leaf)

**Files:**
- Create: `src/hydra_suite/core/identity/pose/vitpose/adapter.py`
- Test: `tests/test_vitpose_adapter.py`

**Interfaces:**
- Consumes: `build_vitpose`, `CheckpointKeyError` (leaf); `torch`.
- Produces: `load_finetuned_checkpoint(path: Path) -> tuple[ViTPose, FinetuneMeta]` where `FinetuneMeta` is a frozen dataclass `(variant: str, head: str, num_keypoints: int)`. Also `infer_head_from_state(state: dict) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_adapter.py
from pathlib import Path

import torch

from hydra_suite.core.identity.pose.vitpose.adapter import (
    FinetuneMeta,
    infer_head_from_state,
    load_finetuned_checkpoint,
)
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose


def _save_training_ckpt(tmp_path: Path, variant: str, head: str, k: int) -> Path:
    model = build_vitpose(variant, head, num_keypoints=k)
    ckpt = {
        "model_state": model.state_dict(),
        "optim_state": {},
        "variant": variant,
        "num_keypoints": k,
        "epoch": 3,
        "pck": 0.5,
        "sched_state": {},
    }
    p = tmp_path / "best.pt"
    torch.save(ckpt, p)
    return p


def test_infer_head_classic_vs_simple(tmp_path):
    classic = build_vitpose("B", "classic", num_keypoints=6).state_dict()
    simple = build_vitpose("B", "simple", num_keypoints=6).state_dict()
    assert infer_head_from_state(classic) == "classic"
    assert infer_head_from_state(simple) == "simple"


def test_load_finetuned_roundtrip(tmp_path):
    p = _save_training_ckpt(tmp_path, "B", "classic", 6)
    model, meta = load_finetuned_checkpoint(p)
    assert isinstance(meta, FinetuneMeta)
    assert meta.variant == "B"
    assert meta.head == "classic"
    assert meta.num_keypoints == 6
    # forward produces (1, 6, 64, 48) heatmaps
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 256, 192))
    assert out.shape == (1, 6, 64, 48)


def test_load_plain_state_dict_infers(tmp_path):
    # a bare state_dict (user-supplied, no metadata wrapper)
    model = build_vitpose("S", "simple", num_keypoints=4)
    p = tmp_path / "plain.pt"
    torch.save(model.state_dict(), p)
    loaded, meta = load_finetuned_checkpoint(p)
    assert meta.head == "simple"
    assert meta.num_keypoints == 4
    assert meta.variant == "S"  # inferred from embed_dim
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: ...vitpose.adapter`.

- [ ] **Step 3: Write the implementation**

```python
# src/hydra_suite/core/identity/pose/vitpose/adapter.py
"""Load a fine-tuned or user-supplied ViTPose checkpoint, recovering the
variant/head/num_keypoints needed to rebuild the module.

PoseKit-free leaf module: imports nothing from hydra_suite app layers.

Bridges the training payload's checkpoint format
(``{"model_state", "variant", "num_keypoints", ...}``) into a live ``ViTPose``.
The leaf ``load_checkpoint`` expects a ``"state_dict"`` key, which the training
format does not have -- hence this adapter. Head type is not stored by the
trainer (it always builds ``"classic"``), so we infer it from parameter shapes,
which also lets arbitrary user checkpoints load.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch

from .config import VARIANTS
from .vitpose import ViTPose, build_vitpose
from .weights import CheckpointKeyError


@dataclass(frozen=True)
class FinetuneMeta:
    variant: str
    head: str
    num_keypoints: int


def _unwrap_state(blob: object) -> Dict[str, torch.Tensor]:
    if isinstance(blob, dict) and "model_state" in blob:
        return blob["model_state"]
    if isinstance(blob, dict) and "state_dict" in blob:
        return blob["state_dict"]
    if isinstance(blob, dict):
        return blob  # bare state_dict
    raise CheckpointKeyError(f"Unrecognized checkpoint object: {type(blob)!r}")


def infer_head_from_state(state: Dict[str, torch.Tensor]) -> str:
    """classic head has deconv layers; simple head does not."""
    has_deconv = any(k.startswith("keypoint_head.deconv_layers.") for k in state)
    return "classic" if has_deconv else "simple"


def _infer_num_keypoints(state: Dict[str, torch.Tensor]) -> int:
    w = state.get("keypoint_head.final_layer.weight")
    if w is None:
        raise CheckpointKeyError(
            "checkpoint has no keypoint_head.final_layer.weight; cannot infer K"
        )
    return int(w.shape[0])


def _infer_variant(state: Dict[str, torch.Tensor]) -> str:
    # embed_dim is the pos_embed last dim: backbone.pos_embed (1, N+1, embed_dim)
    pe = state.get("backbone.pos_embed")
    if pe is None:
        raise CheckpointKeyError(
            "checkpoint has no backbone.pos_embed; cannot infer variant"
        )
    embed_dim = int(pe.shape[-1])
    for name, spec in VARIANTS.items():
        if spec.embed_dim == embed_dim:
            return name
    raise CheckpointKeyError(f"no known variant with embed_dim={embed_dim}")


def load_finetuned_checkpoint(path: Path) -> tuple[ViTPose, FinetuneMeta]:
    path = Path(path)
    blob = torch.load(path, map_location="cpu", weights_only=True)
    state = _unwrap_state(blob)
    head = infer_head_from_state(state)
    if isinstance(blob, dict) and "variant" in blob and "num_keypoints" in blob:
        variant = str(blob["variant"])
        num_keypoints = int(blob["num_keypoints"])
    else:
        variant = _infer_variant(state)
        num_keypoints = _infer_num_keypoints(state)
    model = build_vitpose(variant, head, num_keypoints=num_keypoints)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise CheckpointKeyError(
            f"checkpoint load mismatch: {len(missing)} missing "
            f"{sorted(missing)[:6]}, {len(unexpected)} unexpected "
            f"{sorted(unexpected)[:6]}"
        )
    model.eval()
    return model, FinetuneMeta(variant=variant, head=head, num_keypoints=num_keypoints)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_adapter.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Format + commit**

```bash
conda run -n hydra-mps black src/hydra_suite/core/identity/pose/vitpose/adapter.py tests/test_vitpose_adapter.py
conda run -n hydra-mps isort src/hydra_suite/core/identity/pose/vitpose/adapter.py tests/test_vitpose_adapter.py
git add src/hydra_suite/core/identity/pose/vitpose/adapter.py tests/test_vitpose_adapter.py
git commit -m "feat(vitpose): fine-tuned checkpoint adapter with head inference"
```

## Task 2: Pure crop→keypoints inference driver (leaf)

**Files:**
- Create: `src/hydra_suite/core/identity/pose/vitpose/infer.py`
- Test: `tests/test_vitpose_infer.py`

**Interfaces:**
- Consumes: `box2cs`, `top_down_affine`, `normalize`, `transform_preds` (transforms); `decode_udp_torch` (decode); `IMAGE_SIZE_WH`, `HEATMAP_SIZE_WH` (config).
- Produces: `preprocess_crop(crop_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returning `(chw_float32, center, scale)`; `decode_and_project(heatmaps: torch.Tensor, centers: np.ndarray, scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]` returning `(coords_xy (B,K,2) image-space, maxvals (B,K,1))`. A `forward_fn` callable `(np.ndarray (B,3,256,192)) -> torch.Tensor (B,K,64,48)` is injected by the backend so this module stays runtime-agnostic.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_infer.py
import numpy as np
import torch

from hydra_suite.core.identity.pose.vitpose.infer import (
    decode_and_project,
    preprocess_crop,
)


def test_preprocess_crop_shapes():
    crop = np.zeros((80, 60, 3), dtype=np.uint8)
    chw, center, scale = preprocess_crop(crop)
    assert chw.shape == (3, 256, 192)
    assert chw.dtype == np.float32
    assert center.shape == (2,)
    assert scale.shape == (2,)


def test_decode_and_project_center_peak():
    # a single hot pixel at heatmap center should map near the crop center
    B, K, H, W = 1, 3, 64, 48
    hm = torch.zeros(B, K, H, W)
    hm[:, :, H // 2, W // 2] = 10.0
    centers = np.array([[30.0, 40.0]], dtype=np.float32)  # (B,2)
    scales = np.array([[0.6, 0.8]], dtype=np.float32)  # (B,2) PIXEL_STD units
    coords, maxvals = decode_and_project(hm, centers, scales)
    assert coords.shape == (B, K, 2)
    assert maxvals.shape == (B, K, 1)
    # projected point lands within the crop bbox around the center
    assert np.all(np.isfinite(coords))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_infer.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

```python
# src/hydra_suite/core/identity/pose/vitpose/infer.py
"""Pure crop -> keypoints composition for ViTPose. Runtime-agnostic: the caller
injects a forward function (torch module, ONNX session, TRT engine, or CoreML).

PoseKit-free leaf module.
"""
from __future__ import annotations

import numpy as np
import torch

from .config import HEATMAP_SIZE_WH
from .decode import decode_udp_torch
from .transforms import box2cs, normalize, top_down_affine, transform_preds


def preprocess_crop(crop_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A crop (already the animal's bbox) -> (CHW float32, center, scale).

    The crop's own extent is the box; box2cs applies PADDING_FACTOR + aspect fix.
    """
    h, w = crop_bgr.shape[:2]
    box_xywh = np.array([0.0, 0.0, float(w), float(h)], dtype=np.float32)
    center, scale = box2cs(box_xywh)
    warped = top_down_affine(crop_bgr, center, scale, rot=0.0)
    chw = normalize(warped)
    return chw, center, scale


def decode_and_project(
    heatmaps: torch.Tensor,
    centers: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode heatmaps on-device, then project each set back to image coords.

    heatmaps: (B, K, 64, 48). centers/scales: (B, 2). Returns
    coords (B, K, 2) in image space and maxvals (B, K, 1).
    """
    coords_t, maxvals_t = decode_udp_torch(heatmaps)  # on heatmaps.device
    coords = coords_t.detach().cpu().numpy()
    maxvals = maxvals_t.detach().cpu().numpy()
    out = np.empty_like(coords)
    for i in range(coords.shape[0]):
        out[i] = transform_preds(
            coords[i], centers[i], scales[i], HEATMAP_SIZE_WH
        )
    return out, maxvals
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_infer.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Format + commit**

```bash
conda run -n hydra-mps black src/hydra_suite/core/identity/pose/vitpose/infer.py tests/test_vitpose_infer.py
conda run -n hydra-mps isort src/hydra_suite/core/identity/pose/vitpose/infer.py tests/test_vitpose_infer.py
git add src/hydra_suite/core/identity/pose/vitpose/infer.py tests/test_vitpose_infer.py
git commit -m "feat(vitpose): pure crop->keypoints inference driver"
```

## Task 3: `ViTPoseBackend` native path (Protocol impl)

**Files:**
- Create: `src/hydra_suite/core/identity/pose/backends/vitpose.py`
- Test: `tests/test_vitpose_backend_native.py`

**Interfaces:**
- Consumes: `load_finetuned_checkpoint`/`FinetuneMeta` (Task 1); `preprocess_crop`, `decode_and_project` (Task 2); `PoseResult`, `summarize_keypoints`/`empty_pose_result` (pose package); `IMAGE_SIZE_WH`.
- Produces: `ViTPoseBackend(model_path: str, device: str = "auto", runtime_flavor: str = "native", min_valid_conf: float = 0.2, keypoint_names: Optional[Sequence[str]] = None, batch_size: int = 4)` implementing `PoseInferenceBackend`. `preferred_input_size` returns `256`.

- [ ] **Step 1: Write the failing test** (native path, cpu)

```python
# tests/test_vitpose_backend_native.py
from pathlib import Path

import numpy as np
import torch

from hydra_suite.core.identity.pose.backends.vitpose import ViTPoseBackend
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose


def _ckpt(tmp_path: Path, k: int = 4) -> Path:
    model = build_vitpose("S", "classic", num_keypoints=k)
    torch.save(
        {"model_state": model.state_dict(), "variant": "S", "num_keypoints": k},
        tmp_path / "m.pt",
    )
    return tmp_path / "m.pt"


def test_native_predict_batch_shapes(tmp_path):
    path = _ckpt(tmp_path, k=4)
    be = ViTPoseBackend(
        str(path), device="cpu", keypoint_names=["a", "b", "c", "d"]
    )
    be.warmup()
    crops = [np.zeros((70, 50, 3), np.uint8), np.zeros((90, 60, 3), np.uint8)]
    results = be.predict_batch(crops)
    assert len(results) == 2
    assert results[0].num_keypoints == 4
    assert results[0].keypoints.shape == (4, 3)  # x, y, conf
    assert be.preferred_input_size == 256
    assert be.output_keypoint_names == ["a", "b", "c", "d"]
    be.close()


def test_native_empty_batch(tmp_path):
    be = ViTPoseBackend(str(_ckpt(tmp_path)), device="cpu")
    assert be.predict_batch([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_backend_native.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation** (native path only; accelerated paths added in Phase 2)

```python
# src/hydra_suite/core/identity/pose/backends/vitpose.py
"""ViTPose pose backend. Thin driver over the Spec-1 leaf, mirroring yolo.py.

Native path only in Phase 1; ONNX/TensorRT/CoreML runners are wired in Phase 2.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from ..types import PoseResult
from ..utils import empty_pose_result, summarize_keypoints
from ..vitpose.adapter import load_finetuned_checkpoint
from ..vitpose.config import IMAGE_SIZE_WH
from ..vitpose.infer import decode_and_project, preprocess_crop


def _resolve_device(device: str) -> str:
    if device not in ("auto", ""):
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ViTPoseBackend:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        runtime_flavor: str = "native",
        min_valid_conf: float = 0.2,
        keypoint_names: Optional[Sequence[str]] = None,
        batch_size: int = 4,
    ) -> None:
        self._device = _resolve_device(device)
        self._runtime_flavor = runtime_flavor
        self._min_valid_conf = float(min_valid_conf)
        self._batch_size = int(batch_size)
        model, meta = load_finetuned_checkpoint(Path(model_path))
        self._meta = meta
        self._num_keypoints = meta.num_keypoints
        self.output_keypoint_names: List[str] = list(
            keypoint_names or [f"kp{i}" for i in range(meta.num_keypoints)]
        )
        if len(self.output_keypoint_names) != meta.num_keypoints:
            raise ValueError(
                f"keypoint_names has {len(self.output_keypoint_names)} entries "
                f"but checkpoint has {meta.num_keypoints} keypoints"
            )
        self._model = model.to(self._device).eval()

    @property
    def preferred_input_size(self) -> int:
        return IMAGE_SIZE_WH[1]  # 256 (H); the long side

    def warmup(self) -> None:
        dummy = np.zeros((32, 32, 3), dtype=np.uint8)
        try:
            self.predict_batch([dummy])
        except Exception:  # warmup must never raise
            pass

    def _forward_torch(self, batch_chw: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(batch_chw).to(self._device)
        with torch.no_grad():
            return self._model(t)  # (B, K, 64, 48) on device

    def predict_batch(self, crops: Sequence[np.ndarray]) -> List[PoseResult]:
        if not crops:
            return []
        results: List[PoseResult] = []
        for start in range(0, len(crops), self._batch_size):
            chunk = crops[start : start + self._batch_size]
            chws, centers, scales = [], [], []
            for crop in chunk:
                chw, c, s = preprocess_crop(np.asarray(crop))
                chws.append(chw)
                centers.append(c)
                scales.append(s)
            batch = np.stack(chws, axis=0).astype(np.float32)
            heatmaps = self._forward_torch(batch)
            coords, maxvals = decode_and_project(
                heatmaps, np.stack(centers), np.stack(scales)
            )
            for i in range(len(chunk)):
                kpts = np.concatenate([coords[i], maxvals[i]], axis=1)  # (K,3)
                results.append(
                    summarize_keypoints(kpts, self._min_valid_conf)
                )
        return results

    def close(self) -> None:
        self._model = None
```

**Note for implementer:** verify the exact signature of `summarize_keypoints`/`empty_pose_result` in `core/identity/pose/utils.py` (grep first). If `summarize_keypoints` expects a different argument order or a `num_keypoints` kwarg, adapt the call — the contract is: given a `(K,3)` xy+conf array and a confidence threshold, return a `PoseResult`. `empty_pose_result` is imported for the (unused-in-Phase-1) degenerate path; drop the import if unused to keep output pristine.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_backend_native.py -v`
Expected: PASS (2 tests). Output pristine.

- [ ] **Step 5: Format + commit**

```bash
conda run -n hydra-mps black src/hydra_suite/core/identity/pose/backends/vitpose.py tests/test_vitpose_backend_native.py
conda run -n hydra-mps isort src/hydra_suite/core/identity/pose/backends/vitpose.py tests/test_vitpose_backend_native.py
git add src/hydra_suite/core/identity/pose/backends/vitpose.py tests/test_vitpose_backend_native.py
git commit -m "feat(vitpose): native ViTPoseBackend (Protocol impl)"
```

## Task 4: `PoseRuntimeConfig` fields + factory branch

**Files:**
- Modify: `src/hydra_suite/core/identity/pose/types.py:22-44` (add `vitpose_*` fields)
- Modify: `src/hydra_suite/core/identity/pose/api.py` (add `vitpose` branch before `:159` raise)
- Test: `tests/test_vitpose_factory.py`

**Interfaces:**
- Consumes: `ViTPoseBackend` (Task 3).
- Produces: `create_pose_backend_from_config(config)` returns a `ViTPoseBackend` when `config.backend_family == "vitpose"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_factory.py
from pathlib import Path

import torch

from hydra_suite.core.identity.pose.api import create_pose_backend_from_config
from hydra_suite.core.identity.pose.backends.vitpose import ViTPoseBackend
from hydra_suite.core.identity.pose.types import PoseRuntimeConfig
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose


def test_factory_builds_vitpose(tmp_path):
    model = build_vitpose("S", "classic", num_keypoints=3)
    p = tmp_path / "m.pt"
    torch.save({"model_state": model.state_dict(), "variant": "S",
                "num_keypoints": 3}, p)
    cfg = PoseRuntimeConfig(
        backend_family="vitpose",
        runtime_flavor="native",
        device="cpu",
        model_path=str(p),
        keypoint_names=["a", "b", "c"],
        vitpose_batch=2,
    )
    be = create_pose_backend_from_config(cfg)
    assert isinstance(be, ViTPoseBackend)
    assert be.output_keypoint_names == ["a", "b", "c"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_factory.py -v`
Expected: FAIL — `PoseRuntimeConfig` has no `vitpose_batch`; and/or `RuntimeError: Unsupported pose backend family: vitpose`.

- [ ] **Step 3a: Add fields to `PoseRuntimeConfig`** (after the `sleap_*` block at `types.py:42`, before `keypoint_names` at `:43`)

```python
    # --- vitpose-specific (flat, mirrors yolo_/sleap_) ---
    vitpose_batch: int = 4
    vitpose_variant: str = "auto"  # "auto" = infer from checkpoint
    vitpose_num_keypoints: int = 0  # 0 = infer from checkpoint
```

- [ ] **Step 3b: Add the factory branch** in `api.py`, immediately before the terminal `raise RuntimeError(...)` at `:159`:

```python
    if backend_family == "vitpose":
        from .backends.vitpose import ViTPoseBackend

        model_path = str(config.model_path).strip()
        if not model_path:
            raise RuntimeError("ViTPose backend requires a model_path (checkpoint)")
        return ViTPoseBackend(
            model_path=model_path,
            device=effective_device,
            runtime_flavor=runtime_flavor,
            min_valid_conf=config.min_valid_conf,
            keypoint_names=list(config.keypoint_names) or None,
            batch_size=config.vitpose_batch,
        )
```

**Note:** `effective_device` and `runtime_flavor` are the local names the yolo/sleap branches already use in `create_pose_backend_from_config` (see `api.py:33-52`). Reuse them verbatim; do not recompute.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_factory.py -v`
Expected: PASS.

- [ ] **Step 5: Format + commit**

```bash
conda run -n hydra-mps black src/hydra_suite/core/identity/pose/types.py src/hydra_suite/core/identity/pose/api.py tests/test_vitpose_factory.py
conda run -n hydra-mps isort src/hydra_suite/core/identity/pose/types.py src/hydra_suite/core/identity/pose/api.py tests/test_vitpose_factory.py
git add -A
git commit -m "feat(vitpose): PoseRuntimeConfig fields + factory registry branch"
```

## Task 5: `PoseConfig` sub-config + cache key

**Files:**
- Modify: `src/hydra_suite/core/inference/config.py` (add `PoseViTPoseConfig`; extend `PoseConfig.backend` Literal at `:196`; add `vitpose` field; pop `vitpose` in `_dict_to_config` at `:346-358`)
- Modify: `src/hydra_suite/core/inference/cache/keys.py:166-182` (add `vitpose` branch)
- Test: `tests/test_vitpose_pose_config.py`

**Interfaces:**
- Produces: `PoseViTPoseConfig(model_path: str, variant: str = "auto", num_keypoints: int = 0, batch_size: int = 4)`; `PoseConfig.backend: Literal["yolo","sleap","vitpose"]`, `PoseConfig.vitpose: PoseViTPoseConfig | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_pose_config.py
from hydra_suite.core.inference.config import (
    PoseConfig,
    PoseViTPoseConfig,
    _dict_to_config,
)
from hydra_suite.core.inference.cache.keys import pose_cache_key


def test_posevitposeconfig_roundtrip():
    cfg = PoseConfig(
        backend="vitpose",
        vitpose=PoseViTPoseConfig(model_path="/tmp/best.pt", variant="B",
                                  num_keypoints=6, batch_size=8),
    )
    d = cfg.to_dict() if hasattr(cfg, "to_dict") else None
    assert cfg.backend == "vitpose"
    assert cfg.vitpose.model_path == "/tmp/best.pt"


def test_cache_key_vitpose_branch(tmp_path):
    p = tmp_path / "best.pt"
    p.write_bytes(b"x")
    cfg = PoseConfig(
        backend="vitpose",
        vitpose=PoseViTPoseConfig(model_path=str(p)),
    )
    key = pose_cache_key(cfg)
    assert key.model_path == str(p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_pose_config.py -v`
Expected: FAIL — `PoseViTPoseConfig` undefined.

- [ ] **Step 3a: Add `PoseViTPoseConfig`** after `PoseSLEAPConfig` (`config.py:187`):

```python
@dataclass
class PoseViTPoseConfig:
    model_path: str
    variant: str = "auto"
    num_keypoints: int = 0
    batch_size: int = 4
```

- [ ] **Step 3b: Extend `PoseConfig`** — change `:196` and add the field:

```python
    backend: Literal["yolo", "sleap", "vitpose"] = "yolo"
    skeleton_file: str = ""
    yolo: PoseYOLOConfig | None = None
    sleap: PoseSLEAPConfig | None = None
    vitpose: PoseViTPoseConfig | None = None
```

- [ ] **Step 3c: Pop the sub-dict** in `_dict_to_config` (mirror the yolo/sleap pops at `config.py:346-358`):

```python
    vitpose_d = pose_d.pop("vitpose", None)
    # ... alongside the existing yolo_d / sleap_d handling ...
    vitpose=PoseViTPoseConfig(**vitpose_d) if vitpose_d else None,
```

- [ ] **Step 3d: Add the cache-key branch** in `keys.py` (replace the `if/else` at `:167-172`):

```python
    if config.backend == "yolo":
        assert config.yolo is not None
        path = config.yolo.model_path
    elif config.backend == "vitpose":
        assert config.vitpose is not None
        path = config.vitpose.model_path
    else:
        assert config.sleap is not None
        path = config.sleap.model_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_pose_config.py -v`
Expected: PASS.

- [ ] **Step 5: Format + commit**

```bash
conda run -n hydra-mps black src/hydra_suite/core/inference/config.py src/hydra_suite/core/inference/cache/keys.py tests/test_vitpose_pose_config.py
conda run -n hydra-mps isort src/hydra_suite/core/inference/config.py src/hydra_suite/core/inference/cache/keys.py tests/test_vitpose_pose_config.py
git add -A
git commit -m "feat(vitpose): PoseConfig sub-config + vitpose cache-key branch"
```

## Task 6: Resolver stage + `stages/pose.py` translation branch

**Files:**
- Modify: `src/hydra_suite/runtime/resolver.py:15` (add `"vitpose_pose"` to `STAGES`)
- Modify: `src/hydra_suite/core/inference/stages/pose.py` (add a `vitpose` branch building a `PoseRuntimeConfig` from `resolved`, mirroring the SLEAP branch at `:141-182`; select the stage name `"vitpose_pose"` when `config.backend == "vitpose"`)
- Test: `tests/test_vitpose_stage.py`

**Interfaces:**
- Consumes: `resolved_backend_for`, `create_pose_backend_from_config`, `PoseViTPoseConfig`.
- Produces: the pose stage constructs a `ViTPoseBackend` when `PoseConfig.backend == "vitpose"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_stage.py
from hydra_suite.runtime.resolver import STAGES, RuntimeResolver, PlatformInfo


def test_vitpose_pose_stage_registered():
    assert "vitpose_pose" in STAGES


def test_resolver_vitpose_pose_native_on_cpu():
    r = RuntimeResolver("cpu", PlatformInfo(has_cuda=False, has_mps=False))
    resolved = r.resolve("vitpose_pose")
    assert resolved.backend == "torch"
    assert resolved.device == "cpu"


def test_resolver_vitpose_pose_gpu_fast_cuda():
    r = RuntimeResolver("gpu_fast", PlatformInfo(has_cuda=True, has_mps=False))
    resolved = r.resolve("vitpose_pose", artifact_available=lambda: True)
    assert resolved.backend == "tensorrt"
    assert resolved.device == "cuda"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_stage.py -v`
Expected: FAIL — `"vitpose_pose"` not in `STAGES`.

- [ ] **Step 3a: Add the stage** — `resolver.py:15`:

```python
STAGES = ("obb", "head_tail", "cnn", "yolo_pose", "sleap_pose", "vitpose_pose", "bgsub")
```

(The resolver logic in `resolve()` is stage-agnostic for network stages — only `"bgsub"` is special-cased — so `"vitpose_pose"` inherits the correct torch/tensorrt/coreml resolution with no further change. The tests above confirm this.)

- [ ] **Step 3b: Add the translation branch** in `stages/pose.py`. After `resolved = resolved_backend_for(runtime)` (`:98`), add a `vitpose` branch modeled on the SLEAP branch (`:141-182`). It maps `resolved` → `(device, runtime_flavor)` and builds a `PoseRuntimeConfig`:

```python
    elif config.backend == "vitpose":
        assert config.vitpose is not None
        if resolved.backend == "tensorrt":
            vp_flavor, vp_device = "tensorrt", "cuda"
        elif resolved.backend == "coreml":
            vp_flavor, vp_device = "coreml", "mps"
        else:  # torch
            vp_flavor, vp_device = "native", resolved.device
        runtime_cfg = PoseRuntimeConfig(
            backend_family="vitpose",
            runtime_flavor=vp_flavor,
            device=vp_device,
            model_path=config.vitpose.model_path,
            min_valid_conf=config.min_keypoint_confidence,
            keypoint_names=list(keypoint_names),
            vitpose_batch=config.vitpose.batch_size,
            vitpose_variant=config.vitpose.variant,
            vitpose_num_keypoints=config.vitpose.num_keypoints,
        )
        backend = create_pose_backend_from_config(runtime_cfg)
```

**Note:** match the exact local variable names the SLEAP branch uses for `keypoint_names` and how `backend`/`model` is assigned and returned (read `:141-190`). Do not invent new plumbing — slot the branch into the existing if/elif and return path. Also add `"vitpose_pose"` where the stage picks its resolver stage name (grep for `"sleap_pose"` / `"yolo_pose"` in this file and add the parallel `vitpose` case).

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_stage.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Format + commit**

```bash
conda run -n hydra-mps black src/hydra_suite/runtime/resolver.py src/hydra_suite/core/inference/stages/pose.py tests/test_vitpose_stage.py
conda run -n hydra-mps isort src/hydra_suite/runtime/resolver.py src/hydra_suite/core/inference/stages/pose.py tests/test_vitpose_stage.py
git add -A
git commit -m "feat(vitpose): resolver stage + pose-stage translation branch"
```

## Task 7: GUI family pickers (posekit + trackerkit)

**Files:**
- Modify: `src/hydra_suite/posekit/gui/main_window.py` (the backend selector `_pred_backend`/model-picker sites — grep `"sleap"` around `:4225,4674`)
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py` (backend routing — grep `backend == "sleap"`)
- Modify: `src/hydra_suite/trackerkit/gui/panels/detection_panel.py` (pose backend combo — grep `pose_backend_family`)
- Test: `tests/test_vitpose_gui_wiring.py`

**Interfaces:**
- Consumes: the `vitpose` family end-to-end from Tasks 4–6.
- Produces: `"vitpose"` is a selectable pose backend in both GUIs; selecting it routes a `.pt` checkpoint through `PoseViTPoseConfig`.

**Note:** This task is GUI plumbing — the reviewer should focus on parity with how `sleap` is threaded, not on new logic. Because Qt widgets are hard to unit-test, the test asserts the *config-building* helper (non-Qt) produces a `vitpose` `PoseConfig`, and a smoke import of both modules.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_gui_wiring.py
import importlib


def test_posekit_main_window_imports():
    importlib.import_module("hydra_suite.posekit.gui.main_window")


def test_trackerkit_config_orchestrator_imports():
    importlib.import_module("hydra_suite.trackerkit.gui.orchestrators.config")


def test_detection_panel_offers_vitpose():
    # The pose-backend option list should include vitpose. Grep the constant
    # the panel builds its combo from; adjust the import to the real symbol.
    from hydra_suite.trackerkit.gui.panels import detection_panel

    src = detection_panel.__file__
    with open(src, "r", encoding="utf-8") as fh:
        text = fh.read()
    assert "vitpose" in text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_gui_wiring.py -v`
Expected: FAIL — `test_detection_panel_offers_vitpose` (no `vitpose` in the panel yet).

- [ ] **Step 3: Implement the pickers.** In each file, find where `"sleap"` / `"yolo"` are offered and threaded, and add the parallel `"vitpose"` case:
  - **detection_panel.py**: add `"ViTPose"` to the pose-backend combo items; map the display label → family string `"vitpose"` (mirror the existing `startswith("sleap")` mapping).
  - **posekit main_window.py**: extend `_pred_backend()` (`:4225`) to return `"vitpose"` for a `"vitpose"`-prefixed selection; extend the model-picker dispatch (`:4674`) so a `vitpose` selection browses for a `.pt` checkpoint; extend `_pred_cache_backend`/visibility toggles as the `sleap` branches do.
  - **trackerkit orchestrators/config.py**: where `backend == "sleap"` builds `PoseSLEAPConfig`, add `backend == "vitpose"` building `PoseViTPoseConfig(model_path=..., variant="auto", num_keypoints=0, batch_size=...)`.

  Keep each change a direct parallel of the `sleap` path. Do not add new config knobs beyond `PoseViTPoseConfig`.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_gui_wiring.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Format + commit**

```bash
conda run -n hydra-mps black <changed gui files> tests/test_vitpose_gui_wiring.py
conda run -n hydra-mps isort <changed gui files> tests/test_vitpose_gui_wiring.py
git add -A
git commit -m "feat(vitpose): GUI family pickers (posekit + trackerkit)"
```

**Phase 1 checkpoint:** ViTPose is now selectable and runs natively (cpu/gpu/mps) end-to-end. A fine-tuned `best.pt` can be tracked. Run `conda run -n hydra-mps python -m pytest tests/ -k vitpose -v` — all Phase-1 vitpose tests green.

---

# Phase 2 — Accelerated runtimes (TensorRT / CoreML)

Delivers: `gpu_fast` tier runs ViTPose via a native TensorRT engine (CUDA) or CoreML `.mlpackage` (Apple), built lazily and cached. This phase extracts the shared runners out of `sleap.py` (moving, not duplicating) and wires them into `ViTPoseBackend`.

## Task 8: Extract the shared pose runtime from `sleap.py`

**Files:**
- Create: `src/hydra_suite/core/identity/pose/runtime/__init__.py`
- Create: `src/hydra_suite/core/identity/pose/runtime/onnx_session.py`
- Create: `src/hydra_suite/core/identity/pose/runtime/tensorrt_engine.py`
- Create: `src/hydra_suite/core/identity/pose/runtime/accelerated.py`
- Modify: `src/hydra_suite/core/identity/pose/backends/sleap.py` (delete the moved classes; import them from `..runtime`)
- Test: `tests/test_pose_runtime_extraction.py`

**Interfaces (produced):**
- `onnx_session.OnnxSessionRunner(model_path: Path, resolved: ResolvedBackend)` with `run(batch: np.ndarray) -> Dict[str, np.ndarray]` (and the input-spec detect helpers it needs).
- `tensorrt_engine.TensorRTEngineRunner(model_path: Path)` with `run(batch)` and `run_cuda(batch_cuda)`; `build_trt_engine_from_onnx(onnx_path, engine_path, workspace_gb=4.0, fixed_hw=None) -> bool`.
- `accelerated.build_accelerated_runner(model_path: Path, resolved: ResolvedBackend) -> OnnxSessionRunner | TensorRTEngineRunner` — the suffix-branching ladder (moved from `_init_tensorrt_runner`/`_ort_trt_ep_fallback`).

**Method:** This is a *move*, keeping SLEAP behavior identical. Move the class/function bodies verbatim from the cited `sleap.py` lines into the new modules, renaming `_DirectOnnxSession`→`OnnxSessionRunner`, `_DirectTensorRTEngine`→`TensorRTEngineRunner`, `_build_trt_engine_from_onnx`→`build_trt_engine_from_onnx`, `_init_tensorrt_runner`(+`_ort_trt_ep_fallback`)→`build_accelerated_runner`. Then in `sleap.py`, delete the moved bodies and add `from ..runtime.onnx_session import OnnxSessionRunner` etc., aliasing internally if needed (`_DirectOnnxSession = OnnxSessionRunner`) so the rest of `sleap.py` is untouched.

Move source ranges (verbatim): `_DirectOnnxSession` `sleap.py:368-390` + the ONNX input-spec helpers it references (grep `_detect_onnx_*` near `:139-218`); `_DirectTensorRTEngine` `:513-762`; `_build_trt_engine_from_onnx` `:399-511`; ladder `:838-946`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pose_runtime_extraction.py
import numpy as np
import pytest


def test_onnx_runner_runs_tiny_model(tmp_path):
    pytest.importorskip("onnxruntime")
    import onnx
    from onnx import helper, TensorProto

    from hydra_suite.core.identity.pose.runtime.onnx_session import OnnxSessionRunner
    from hydra_suite.runtime.resolver import ResolvedBackend

    # identity ONNX: input (1,3,4,4) -> output same
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "id", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    p = tmp_path / "id.onnx"
    onnx.save(model, str(p))

    runner = OnnxSessionRunner(p, ResolvedBackend("torch", "cpu", False))
    out = runner.run(np.zeros((1, 3, 4, 4), np.float32))
    # returns a dict of output name -> array
    arr = next(iter(out.values())) if isinstance(out, dict) else out[0]
    assert np.asarray(arr).shape == (1, 3, 4, 4)


def test_sleap_still_imports_after_extraction():
    # SLEAP must keep working: its module imports the moved runners
    import importlib

    importlib.import_module("hydra_suite.core.identity.pose.backends.sleap")
    from hydra_suite.core.identity.pose.runtime.onnx_session import OnnxSessionRunner
    from hydra_suite.core.identity.pose.runtime.tensorrt_engine import (
        TensorRTEngineRunner,
        build_trt_engine_from_onnx,
    )
    from hydra_suite.core.identity.pose.runtime.accelerated import (
        build_accelerated_runner,
    )
    assert OnnxSessionRunner and TensorRTEngineRunner
    assert build_trt_engine_from_onnx and build_accelerated_runner
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_pose_runtime_extraction.py -v`
Expected: FAIL — `hydra_suite...pose.runtime.onnx_session` not found.

- [ ] **Step 3: Perform the move.** Create the four modules with the moved bodies (adjust imports: `execution_providers_for` from `hydra_suite.runtime.onnx_providers`, `ResolvedBackend` from `hydra_suite.runtime.resolver`, numpy/torch/tensorrt lazily as the originals do). Repoint `sleap.py` with import aliases. Keep the `run`/`run_cuda`/`build`/ladder logic byte-identical.

- [ ] **Step 4: Run the extraction test + the existing SLEAP suite**

Run: `conda run -n hydra-mps python -m pytest tests/test_pose_runtime_extraction.py -v && conda run -n hydra-mps python -m pytest tests/ -k sleap -v`
Expected: extraction PASS; SLEAP tests unchanged from baseline (same pass/skip counts).

**Pre-merge manual gate (name it in the report, do not skip):** if SLEAP export artifacts are available in the environment, run `python tools/equivalence/verify_sleap_exported_vs_service.py` and confirm keypoint parity is unchanged. If artifacts are unavailable, state that explicitly in the report — the reviewer decides whether to require it before merge.

- [ ] **Step 5: Format + commit**

```bash
conda run -n hydra-mps black src/hydra_suite/core/identity/pose/runtime/*.py src/hydra_suite/core/identity/pose/backends/sleap.py tests/test_pose_runtime_extraction.py
conda run -n hydra-mps isort src/hydra_suite/core/identity/pose/runtime/*.py src/hydra_suite/core/identity/pose/backends/sleap.py tests/test_pose_runtime_extraction.py
git add -A
git commit -m "refactor(pose): extract shared runtime (onnx/tensorrt/ladder) from sleap.py"
```

## Task 9: `CoreMLRunner` (new)

**Files:**
- Create: `src/hydra_suite/core/identity/pose/runtime/coreml_runner.py`
- Test: `tests/test_coreml_runner.py`

**Interfaces:**
- Produces: `CoreMLRunner(model_path: Path)` with `run(batch: np.ndarray) -> Dict[str, np.ndarray]`. Honors the leaf export's static batch=1 by looping per sample.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coreml_runner.py
import numpy as np
import pytest


def test_coreml_runner_signature_and_import():
    from hydra_suite.core.identity.pose.runtime.coreml_runner import CoreMLRunner

    assert hasattr(CoreMLRunner, "run")


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("coremltools") is None,
    reason="coremltools not installed",
)
def test_coreml_runner_predicts(tmp_path):
    ct = pytest.importorskip("coremltools")
    import torch

    from hydra_suite.core.identity.pose.runtime.coreml_runner import CoreMLRunner
    from hydra_suite.core.identity.pose.vitpose.export import export_coreml
    from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose

    model = build_vitpose("S", "classic", num_keypoints=3).eval()
    path = tmp_path / "m.mlpackage"
    export_coreml(model, path)
    runner = CoreMLRunner(path)
    out = runner.run(np.zeros((1, 3, 256, 192), np.float32))
    arr = next(iter(out.values())) if isinstance(out, dict) else out
    assert np.asarray(arr).shape[0] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_coreml_runner.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

```python
# src/hydra_suite/core/identity/pose/runtime/coreml_runner.py
"""Native CoreML .mlpackage runner for pose backends.

The leaf export pins CoreML input to a static batch of 1 (pos_embed has no
interpolation path), so this runner loops per sample. fp32 throughout.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


class CoreMLRunner:
    def __init__(self, model_path: Path) -> None:
        import coremltools as ct  # lazy

        self._model = ct.models.MLModel(str(model_path))
        spec = self._model.get_spec()
        self._input_name = spec.description.input[0].name
        self._output_name = spec.description.output[0].name

    def run(self, batch: np.ndarray) -> Dict[str, np.ndarray]:
        batch = np.asarray(batch, dtype=np.float32)
        outs = []
        for i in range(batch.shape[0]):
            sample = batch[i : i + 1]  # (1,3,256,192)
            pred = self._model.predict({self._input_name: sample})
            outs.append(np.asarray(pred[self._output_name], dtype=np.float32))
        return {self._output_name: np.concatenate(outs, axis=0)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_coreml_runner.py -v`
Expected: PASS (2 tests; the predict test may skip if coremltools absent).

- [ ] **Step 5: Format + commit**

```bash
conda run -n hydra-mps black src/hydra_suite/core/identity/pose/runtime/coreml_runner.py tests/test_coreml_runner.py
conda run -n hydra-mps isort src/hydra_suite/core/identity/pose/runtime/coreml_runner.py tests/test_coreml_runner.py
git add -A
git commit -m "feat(pose): CoreMLRunner for native .mlpackage inference"
```

## Task 10: `auto_export_vitpose_model` caching wrapper

**Files:**
- Modify: `src/hydra_suite/core/identity/pose/backends/vitpose.py` (add module-level `auto_export_vitpose_model`)
- Test: `tests/test_auto_export_vitpose.py`

**Interfaces:**
- Consumes: `load_finetuned_checkpoint` (Task 1); leaf `export_onnx`/`build_tensorrt_engine`/`export_coreml`; `artifacts` primitives.
- Produces: `auto_export_vitpose_model(config: PoseRuntimeConfig, runtime_flavor: str, runtime_device: Optional[str] = None) -> str` — returns the artifact path (`.engine` for tensorrt, `.mlpackage` for coreml), co-located with the checkpoint, signature-gated with recipe tag `"vitpose-v1"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_auto_export_vitpose.py
from pathlib import Path

import pytest
import torch

from hydra_suite.core.identity.pose.backends.vitpose import (
    _vitpose_artifact_signature,
    auto_export_vitpose_model,
)
from hydra_suite.core.identity.pose.types import PoseRuntimeConfig
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose


def _ckpt(tmp_path):
    model = build_vitpose("S", "classic", num_keypoints=3)
    p = tmp_path / "best.pt"
    torch.save({"model_state": model.state_dict(), "variant": "S",
                "num_keypoints": 3}, p)
    return p


def test_signature_includes_recipe_tag(tmp_path):
    p = _ckpt(tmp_path)
    sig = _vitpose_artifact_signature(str(p), "coreml")
    assert "vitpose-v1" in sig
    assert "coreml" in sig


def test_coreml_export_cached(tmp_path):
    pytest.importorskip("coremltools")
    p = _ckpt(tmp_path)
    cfg = PoseRuntimeConfig(backend_family="vitpose", model_path=str(p),
                            runtime_flavor="coreml", device="mps")
    art = auto_export_vitpose_model(cfg, "coreml")
    assert Path(art).exists()
    assert Path(art).suffix == ".mlpackage" or Path(art).name.endswith(".mlpackage")
    # second call reuses (mtime unchanged)
    mtime = Path(art).stat().st_mtime_ns
    art2 = auto_export_vitpose_model(cfg, "coreml")
    assert art2 == art
    assert Path(art).stat().st_mtime_ns == mtime  # not re-exported
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_auto_export_vitpose.py -v`
Expected: FAIL — `auto_export_vitpose_model`/`_vitpose_artifact_signature` undefined.

- [ ] **Step 3: Write the implementation** (append to `backends/vitpose.py`)

```python
# --- appended to backends/vitpose.py ---
from ..artifacts import artifact_meta_matches, path_fingerprint_token, write_artifact_meta
from ..vitpose.export import build_tensorrt_engine, export_coreml, export_onnx

_VITPOSE_RECIPE_TAG = "vitpose-v1"


def _vitpose_artifact_signature(model_path: str, flavor: str) -> str:
    return f"{_VITPOSE_RECIPE_TAG}|{flavor}|opset17|fp32|{path_fingerprint_token(model_path)}"


def _artifact_path_for(model_path: Path, flavor: str) -> Path:
    if flavor == "tensorrt":
        return model_path.with_suffix(".engine")
    if flavor == "coreml":
        return model_path.with_suffix(".mlpackage")
    raise ValueError(f"no ViTPose artifact for flavor {flavor!r}")


def auto_export_vitpose_model(
    config, runtime_flavor: str, runtime_device: Optional[str] = None
) -> str:
    """Lazily export + cache a ViTPose artifact next to its checkpoint.

    Mirrors auto_export_yolo_model / auto_export_sleap_model: co-located
    artifact, signature-gated .runtime_meta.json sidecar, recipe-version tag.
    """
    model_path = Path(str(config.model_path))
    artifact = _artifact_path_for(model_path, runtime_flavor)
    signature = _vitpose_artifact_signature(str(model_path), runtime_flavor)
    if artifact.exists() and artifact_meta_matches(artifact, signature):
        return str(artifact.resolve())

    model, _meta = load_finetuned_checkpoint(model_path)
    model.eval()
    if runtime_flavor == "coreml":
        export_coreml(model, artifact)
    elif runtime_flavor == "tensorrt":
        onnx_path = model_path.with_suffix(".onnx")
        export_onnx(model, onnx_path)
        build_tensorrt_engine(onnx_path, artifact, fp16=False)
    else:
        raise ValueError(f"auto_export_vitpose_model: bad flavor {runtime_flavor!r}")
    write_artifact_meta(artifact, signature)
    return str(artifact.resolve())
```

**Note:** put the new imports at the top of the file with the others (do not leave them mid-file). Shown inline here only to keep the diff readable.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_auto_export_vitpose.py -v`
Expected: PASS (2 tests; the export test skips without coremltools).

- [ ] **Step 5: Format + commit**

```bash
conda run -n hydra-mps black src/hydra_suite/core/identity/pose/backends/vitpose.py tests/test_auto_export_vitpose.py
conda run -n hydra-mps isort src/hydra_suite/core/identity/pose/backends/vitpose.py tests/test_auto_export_vitpose.py
git add -A
git commit -m "feat(vitpose): auto_export_vitpose_model caching wrapper"
```

## Task 11: Wire accelerated runners into `ViTPoseBackend`

**Files:**
- Modify: `src/hydra_suite/core/identity/pose/backends/vitpose.py` (construct a runner in `__init__` for `tensorrt`/`coreml`; route `_forward` through it; add `predict_batch_cuda`)
- Modify: `src/hydra_suite/core/identity/pose/api.py` (in the `vitpose` branch, call `auto_export_vitpose_model` for `tensorrt`/`coreml` flavors and pass the artifact path as `exported_model_path`, mirroring the yolo branch at `api.py:73-97`)
- Test: `tests/test_vitpose_backend_accelerated.py`

**Interfaces:**
- Consumes: `build_accelerated_runner` (Task 8), `CoreMLRunner` (Task 9), `auto_export_vitpose_model` (Task 10).
- Produces: `ViTPoseBackend` runs `tensorrt`/`coreml` flavors; exposes `predict_batch_cuda(crops)` (zero-copy only when the runner is a `TensorRTEngineRunner`; otherwise forwards to `predict_batch`).

- [ ] **Step 1: Write the failing test** (CoreML path if available; else assert the wiring/attributes)

```python
# tests/test_vitpose_backend_accelerated.py
from pathlib import Path

import numpy as np
import pytest
import torch

from hydra_suite.core.identity.pose.backends.vitpose import (
    auto_export_vitpose_model,
    ViTPoseBackend,
)
from hydra_suite.core.identity.pose.types import PoseRuntimeConfig
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose


def _ckpt(tmp_path, k=3):
    model = build_vitpose("S", "classic", num_keypoints=k)
    p = tmp_path / "best.pt"
    torch.save({"model_state": model.state_dict(), "variant": "S",
                "num_keypoints": k}, p)
    return p


def test_predict_batch_cuda_falls_back_to_numpy(tmp_path):
    # On a non-TRT runner, predict_batch_cuda must degrade to predict_batch.
    be = ViTPoseBackend(str(_ckpt(tmp_path)), device="cpu",
                        keypoint_names=["a", "b", "c"])
    assert hasattr(be, "predict_batch_cuda")


def test_coreml_backend_predicts(tmp_path):
    pytest.importorskip("coremltools")
    p = _ckpt(tmp_path, k=3)
    cfg = PoseRuntimeConfig(backend_family="vitpose", model_path=str(p),
                            runtime_flavor="coreml", device="mps")
    art = auto_export_vitpose_model(cfg, "coreml")
    be = ViTPoseBackend(str(p), device="mps", runtime_flavor="coreml",
                        keypoint_names=["a", "b", "c"], exported_model_path=art)
    res = be.predict_batch([np.zeros((60, 40, 3), np.uint8)])
    assert res[0].keypoints.shape == (3, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_backend_accelerated.py -v`
Expected: FAIL — `ViTPoseBackend.__init__` has no `exported_model_path`; no `predict_batch_cuda`.

- [ ] **Step 3: Wire the runners.** Extend `ViTPoseBackend.__init__` with `exported_model_path: str = ""`; when `runtime_flavor in ("tensorrt", "coreml")` construct the runner:

```python
        self._runner = None
        if runtime_flavor == "coreml" and exported_model_path:
            from ..runtime.coreml_runner import CoreMLRunner

            self._runner = CoreMLRunner(Path(exported_model_path))
        elif runtime_flavor == "tensorrt" and exported_model_path:
            from hydra_suite.runtime.resolver import ResolvedBackend

            from ..runtime.accelerated import build_accelerated_runner

            self._runner = build_accelerated_runner(
                Path(exported_model_path), ResolvedBackend("tensorrt", "cuda", False)
            )
```

Route the forward: in `_forward_torch`, if `self._runner is not None`, call `self._runner.run(batch)` and coerce the output dict to a `(B,K,64,48)` torch tensor on CPU (then `decode_and_project` still runs via `decode_udp_torch` on CPU). Add:

```python
    def _forward(self, batch_chw: np.ndarray) -> torch.Tensor:
        if self._runner is not None:
            out = self._runner.run(batch_chw.astype(np.float32))
            arr = next(iter(out.values())) if isinstance(out, dict) else out
            return torch.from_numpy(np.asarray(arr, dtype=np.float32))
        return self._forward_torch(batch_chw)

    def predict_batch_cuda(self, crops):
        # Convert any device tensors back to uint8 HWC numpy and reuse the
        # correct numpy path. This is the shippable, correct implementation
        # for every runner. Zero-copy TRT is a documented perf follow-up.
        np_crops = [
            (c.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            if hasattr(c, "permute")
            else np.asarray(c)
            for c in crops
        ]
        return self.predict_batch(np_crops)
```

Replace `predict_batch`'s `self._forward_torch(batch)` call with `self._forward(batch)`.

**Note for implementer (scope guard):** ship the numpy `predict_batch_cuda` above — it is correct for all runners. Do **not** attempt the TRT zero-copy path in this task; it requires feeding device tensors into `TensorRTEngineRunner.run_cuda` mirroring `sleap.py:1110-1160`, and getting it wrong silently corrupts keypoints. Record "ViTPose TRT zero-copy (`predict_batch_cuda`) not yet implemented — numpy path in use" as a known perf follow-up in your report. Exposing `predict_batch_cuda` at all is what the stage's `hasattr` probe (`pose.py:327`) needs; correctness beats zero-copy here.

- [ ] **Step 3b: Trigger export in `api.py`.** In the `vitpose` factory branch (Task 4), before constructing the backend, when `runtime_flavor in ("tensorrt", "coreml")`, mirror the yolo pattern (`api.py:73-85`):

```python
        exported = ""
        if runtime_flavor in ("tensorrt", "coreml"):
            from .backends.vitpose import auto_export_vitpose_model

            try:
                exported = auto_export_vitpose_model(
                    config, runtime_flavor, runtime_device=effective_device
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ViTPose %s export failed (%s); using native runtime.",
                    runtime_flavor, exc,
                )
                runtime_flavor = "native"
        return ViTPoseBackend(
            model_path=model_path,
            device=effective_device,
            runtime_flavor=runtime_flavor,
            min_valid_conf=config.min_valid_conf,
            keypoint_names=list(config.keypoint_names) or None,
            batch_size=config.vitpose_batch,
            exported_model_path=exported,
        )
```

(Replace the Task-4 `return ViTPoseBackend(...)` with this expanded version. `logger` is the module logger already defined in `api.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_backend_accelerated.py -v`
Expected: PASS (2 tests; coreml test skips without coremltools).

- [ ] **Step 5: Format + commit**

```bash
conda run -n hydra-mps black src/hydra_suite/core/identity/pose/backends/vitpose.py src/hydra_suite/core/identity/pose/api.py tests/test_vitpose_backend_accelerated.py
conda run -n hydra-mps isort src/hydra_suite/core/identity/pose/backends/vitpose.py src/hydra_suite/core/identity/pose/api.py tests/test_vitpose_backend_accelerated.py
git add -A
git commit -m "feat(vitpose): wire TensorRT/CoreML runners + lazy export into backend"
```

**Phase 2 checkpoint:** `gpu_fast` runs ViTPose via native TensorRT (CUDA) / CoreML (Apple), lazily exported and cached; native remains the fallback the resolver selects when no artifact exists.

---

# Phase 3 — Parity + end-to-end

## Task 12: Multi-runtime parity harness

**Files:**
- Create: `tools/equivalence/verify_vitpose_runtimes.py`
- Test: `tests/test_vitpose_parity.py`

**Interfaces:**
- Consumes: `ViTPoseBackend` (native = oracle), `auto_export_vitpose_model`; the leaf `decode_udp_cv2` oracle for the decode check.
- Produces: `compare_runtimes(checkpoint: str, crops: list[np.ndarray], flavors: list[str]) -> dict[str, float]` returning max per-keypoint pixel deviation of each flavor vs the native torch reference.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_parity.py
import numpy as np
import torch

from hydra_suite.core.identity.pose.vitpose.decode import (
    decode_udp_cv2,
    decode_udp_torch,
)


def test_decode_torch_matches_cv2_oracle():
    # the two decoders must agree sub-pixel (the leaf's own gate, re-asserted)
    rng = np.random.default_rng(0)
    hm = rng.random((2, 4, 64, 48)).astype(np.float32)
    c_t, v_t = decode_udp_torch(torch.from_numpy(hm))
    c_c, v_c = decode_udp_cv2(hm)
    assert np.max(np.abs(c_t.numpy() - c_c)) < 0.1  # sub-pixel


def test_parity_harness_importable():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "verify_vitpose_runtimes",
        root / "tools" / "equivalence" / "verify_vitpose_runtimes.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "compare_runtimes")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_parity.py -v`
Expected: FAIL — harness file does not exist.

- [ ] **Step 3: Write the harness**

```python
# tools/equivalence/verify_vitpose_runtimes.py
"""Compare ViTPose runtimes against the native torch reference on real crops.

Native torch is the oracle. Prints max per-keypoint pixel deviation for each
requested flavor. fp32 everywhere, so thresholds are tight:
torch/onnx sub-pixel, tensorrt/coreml <= ~1px.

Usage:
    python tools/equivalence/verify_vitpose_runtimes.py CHECKPOINT CROP_DIR \
        --flavors coreml tensorrt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from hydra_suite.core.identity.pose.backends.vitpose import (
    auto_export_vitpose_model,
    ViTPoseBackend,
)
from hydra_suite.core.identity.pose.types import PoseRuntimeConfig


def _native(checkpoint: str, crops: List[np.ndarray]) -> np.ndarray:
    be = ViTPoseBackend(checkpoint, device="cpu", runtime_flavor="native")
    return np.stack([r.keypoints for r in be.predict_batch(crops)])


def compare_runtimes(
    checkpoint: str, crops: List[np.ndarray], flavors: List[str]
) -> Dict[str, float]:
    ref = _native(checkpoint, crops)
    out: Dict[str, float] = {}
    for flavor in flavors:
        device = {"coreml": "mps", "tensorrt": "cuda"}.get(flavor, "cpu")
        cfg = PoseRuntimeConfig(
            backend_family="vitpose", model_path=checkpoint,
            runtime_flavor=flavor, device=device,
        )
        art = auto_export_vitpose_model(cfg, flavor)
        be = ViTPoseBackend(checkpoint, device=device, runtime_flavor=flavor,
                            exported_model_path=art)
        got = np.stack([r.keypoints for r in be.predict_batch(crops)])
        out[flavor] = float(np.max(np.abs(got[..., :2] - ref[..., :2])))
    return out


def _load_crops(crop_dir: Path) -> List[np.ndarray]:
    import cv2

    return [cv2.imread(str(p)) for p in sorted(crop_dir.glob("*.png"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("crop_dir")
    ap.add_argument("--flavors", nargs="+", default=["coreml"])
    args = ap.parse_args()
    crops = _load_crops(Path(args.crop_dir))
    devs = compare_runtimes(args.checkpoint, crops, args.flavors)
    for flavor, dev in devs.items():
        print(f"{flavor}: max pixel deviation vs native = {dev:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_parity.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Format + commit**

```bash
conda run -n hydra-mps black tools/equivalence/verify_vitpose_runtimes.py tests/test_vitpose_parity.py
conda run -n hydra-mps isort tools/equivalence/verify_vitpose_runtimes.py tests/test_vitpose_parity.py
git add -A
git commit -m "feat(vitpose): multi-runtime parity harness"
```

## Task 13: End-to-end train→track smoke test

**Files:**
- Create: `tests/test_vitpose_e2e_smoke.py`
- Test: (self)

**Interfaces:**
- Consumes: the full stack (Tasks 1–11) + the Spec-4 training payload.

**Purpose:** prove the acceptance loop — a tiny fine-tune produces a `best.pt` that the backend loads and tracks. Kept small (few steps, tiny image) so it runs in CI on CPU.

- [ ] **Step 1: Write the test**

```python
# tests/test_vitpose_e2e_smoke.py
"""Acceptance loop: (mini) train -> best.pt -> backend -> keypoints.

Uses the training payload's own model builder to fabricate a 'trained'
checkpoint (a real training run is covered by Spec-4 tests); this asserts the
integration seam: a training-format checkpoint tracks end to end through the
production factory.
"""
import numpy as np
import torch

from hydra_suite.core.identity.pose.api import create_pose_backend_from_config
from hydra_suite.core.identity.pose.types import PoseRuntimeConfig
from hydra_suite.core.identity.pose.vitpose.training.model_setup import (
    build_finetune_model,
)


def test_training_checkpoint_tracks_end_to_end(tmp_path):
    # 1. produce a training-format best.pt (classic head, as the trainer does)
    model = build_finetune_model(variant="S", num_keypoints=5, drop_path=0.1)
    ckpt = {
        "model_state": model.state_dict(),
        "optim_state": {},
        "variant": "S",
        "num_keypoints": 5,
        "epoch": 1,
        "pck": 0.42,
        "sched_state": {},
    }
    best = tmp_path / "best.pt"
    torch.save(ckpt, best)

    # 2. build the production backend through the factory
    cfg = PoseRuntimeConfig(
        backend_family="vitpose", runtime_flavor="native", device="cpu",
        model_path=str(best), keypoint_names=[f"k{i}" for i in range(5)],
    )
    backend = create_pose_backend_from_config(cfg)

    # 3. track two synthetic crops
    crops = [np.random.randint(0, 255, (80, 60, 3), np.uint8) for _ in range(2)]
    results = backend.predict_batch(crops)
    assert len(results) == 2
    assert results[0].keypoints.shape == (5, 3)
    assert np.all(np.isfinite(results[0].keypoints))
    backend.close()
```

- [ ] **Step 2: Run it (should pass once Phase 1 is complete)**

Run: `conda run -n hydra-mps python -m pytest tests/test_vitpose_e2e_smoke.py -v`
Expected: PASS.

- [ ] **Step 3: Run the full vitpose suite + lint**

```bash
conda run -n hydra-mps python -m pytest tests/ -k "vitpose or pose_runtime or coreml_runner" -v
conda run -n hydra-mps black tests/test_vitpose_e2e_smoke.py
conda run -n hydra-mps isort tests/test_vitpose_e2e_smoke.py
```
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_vitpose_e2e_smoke.py
git commit -m "test(vitpose): end-to-end train-format->track smoke"
```

- [ ] **Step 5: Pre-PR gate**

```bash
conda run -n hydra-mps make lint-moderate
conda run -n hydra-mps make docs-check
```
Fix any findings, commit.

---

## Global Constraints recap for reviewers

Copy the [Global Constraints](#global-constraints) block into each task-reviewer dispatch. The binding, project-specific values: **fp32 only**; **`runtime_flavor ∈ {native, tensorrt, coreml}`** (no `onnx` tier); **`IMAGE_SIZE_WH=(192,256)`**, heatmaps `(B,K,64,48)`; **artifact signature carries `vitpose-v1` recipe tag**; **head is inferred, never stored**; **leaf/Core import no app layers**; **`hydra-mps` env, `black`/`isort` directly (no `make format`)**.

## Dependency graph (for parallel-safe review, not parallel implementation)

```
T1 adapter ─┬─> T3 native backend ─┬─> T4 factory ─> T5 config ─> T6 stage ─> T7 GUI   (Phase 1)
T2 infer  ──┘                      │
                                   └─> T10 export ─┐
T8 runtime extraction ─> T9 coreml ────────────────┴─> T11 wire accel   (Phase 2)
                                          T11 ─> T12 parity ─> T13 e2e   (Phase 3)
```

Implement strictly in numeric order (each task's tests depend on prior tasks). T8/T9 (Phase 2 runtime) have no dependency on T1–T7 and could be built first, but keeping numeric order lets the Phase-1 checkpoint (usable native ViTPose) land and be reviewed before the riskier SLEAP extraction.
