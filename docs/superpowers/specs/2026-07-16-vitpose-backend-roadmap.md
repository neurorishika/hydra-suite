# ViTPose Native Backend — Roadmap

**Date:** 2026-07-16
**Status:** Roadmap agreed. Spec 1 in design.
**Purpose:** Durable context anchor across sessions. This document holds the *why*
and the *ordering*. Each spec below gets its own design doc, plan, and
implementation cycle.

## Goal

Run inference and fine-tune ViTPose models (classic ViTPose B/L/H and ViTPose+
MoE variants) as a first-class pose backend in HYDRA Suite, on CUDA and MPS, with
ONNX/TensorRT export available, and with **no GPU→CPU roundtrips** in the
inference path.

## The core decision: port to native PyTorch, do not use mmpose/mmcv

Decided 2026-07-16. The evidence:

- **No compiled ops in the forward path.** `mmpose/models/backbones/vit.py` and
  `vit_moe.py` import zero mmcv (only `torch` and three `timm.models.layers`
  helpers). Nothing in ViTPose imports `mmcv.ops` or `mmcv._ext`. The only mmcv
  usage in the head is pure-Python layer *factories* (`build_conv_layer`,
  `build_norm_layer`, `build_upsample_layer`) that resolve to plain
  `nn.Conv2d` / `nn.BatchNorm2d` / `nn.ConvTranspose2d`. The detector uses
  `mmcv.image`/`mmcv.visualization` for IO only.
- **The entire mmcv coupling is scaffolding**: config system, registry, runner,
  dataset pipeline. The model itself is ~250 lines of clean PyTorch.
- **mmcv would be actively costly here.** It is unmaintained, and would have to be
  built against the CUDA-12-userspace matrix pinned in `environment-cuda.yml:33-45`.
  There is currently no mmcv/mmpose/openmmlab anywhere in this repo.
- **mmpose cannot meet our requirements anyway.** Its exporter hardcodes
  `assert opset_version == 11`. `torch.compile`/AMP through mmcv is impractical.
- **`timm` is already a base dependency**, so the ViT building blocks are free.
- **HuggingFace `transformers.VitPose` is not a substitute**: it contains a literal
  `raise NotImplementedError("Training is not yet supported")`, converts only 7
  checkpoints (no non-plus S/L/H, no simple decoder beyond base), has no
  flip-test, and no `use_udp` toggle. Useful as a *reference implementation* and
  a *converted-weights source*; not as our training or inference path.

## The three silent-accuracy traps (spine of Spec 1)

These are why a ViTPose port fails quietly rather than loudly. Each is verified
against upstream source.

1. **Patch embed uses `padding=2`, not 0.** `Conv2d(3, D, kernel_size=16,
   stride=16, padding=2)`. Output grid is *coincidentally* the same 16×12 = 192
   tokens, so a stock `timm` patch embed loads with **no shape error** and
   silently samples a shifted pixel grid.
2. **Pos-embed adds the cls slot to every token**:
   `x = x + pos_embed[:, 1:] + pos_embed[:, :1]`. `pos_embed` is
   `(1, num_patches+1, D)` — the MAE cls slot is retained even though no
   `cls_token` module exists. Dropping the second term changes outputs.
3. **UDP is unconditional in every released checkpoint.** Warp, encode, and decode
   must all agree. Mixing UDP decode with non-UDP warp (or vice versa) costs
   ~1–2 AP with nothing visible in the logs.

**Diagnostic ladder** (from parity work): off by ~1 AP → UDP mismatch; ~0.3 AP →
decode blur sigma; wildly off → patch padding or pos-embed.

## Weights

Official links are OneDrive-only (94 `1drv.ms` links in the README; `curl` gets
403 behind a JS/bot gate). **Use `nielsr/vitpose-original-checkpoints` on
HuggingFace instead** — it re-hosts the originals and is what HF's own conversion
script downloads. Contains `vitpose-b.pth`, `vitpose-b-simple.pth`,
`vitpose_base_coco_aic_mpii.pth`, `vitpose+_{small,base,large,huge}.pth`,
`vitpose_small.pth`, `vitpose_small_up4.pth`.

**Gap:** no non-plus L/H there — those still require OneDrive.

Key structure: `torch.load(...)["state_dict"]`, prefixes `backbone.`,
`keypoint_head.`, plus `associate_keypoint_heads.` for ViTPose+.

**Load with `weights_only=True`.** These checkpoints are plain tensor state_dicts,
so this costs nothing — and we are deliberately sourcing them from a *third-party
re-host* rather than upstream, which is exactly the threat model `torch.load`'s
`weights_only=False` default exposes (arbitrary code execution via unpickling).
Record each checkpoint's SHA256 when first downloaded.

## The four specs

Ordering is deliberate and argued below. Each links to its own design doc when written.

### Spec 1 — Standalone ViTPose port + numerical parity
**Status:** in design (`2026-07-16-vitpose-native-port-design.md`)

Backbone (classic + MoE), both heads (classic deconv + simple), weight loader,
pre/post-processing, and (added 2026-07-16) the **export recipe** (`vitpose/export.py`:
torch -> ONNX/TensorRT/CoreML, one parity test per artifact). **Imports nothing from
`hydra_suite`.** No repo integration whatsoever.

The export recipe belongs here because it is model-specific -- it is the piece ultralytics'
`model.export()` supplies for YOLO, which nobody supplies for ViTPose. The auto-export
*caching wrapper* is NOT here; it goes in `backends/vitpose.py` at Spec 3, mirroring
`auto_export_yolo_model`/`auto_export_sleap_model`.

**Done when:** `vitpose-b.pth` and `vitpose+_base.pth` load with `strict=True`,
and COCO val AP reproduces published numbers (75.8 classic / 75.5 simple for B)
within ~0.2. Requires downloading COCO val2017 + standard person detections.

### Spec 2 — Native runtime layer + tensor-first Protocol
**Status:** deferred

Extract a shared native-torch runtime: device resolution, warmup, AMP,
`torch.compile`, batching, ONNX session, TensorRT engine, artifact auto-export.

**Note (2026-07-16):** the export *recipe* moved into Spec 1 (`vitpose/export.py`) —
it is model-specific, and it is the piece ultralytics supplies for YOLO. Spec 2 owns
runtime EXECUTION (sessions, engine lifecycle, warmup/AMP/batching) and consumes
artifacts Spec 1 has already proven exportable and numerically faithful (Gate D).

**Motivated by existing duplication, not speculation.** `sleap.py` is 1780 lines
(against CLAUDE.md's ~500-line rule) and hides `_DirectOnnxSession` (`sleap.py:343`)
and `_DirectTensorRTEngine` (`sleap.py:468`); `core/detectors/_direct_obb_runtime.py`
does morally similar work. Two-to-three real consumers today.

**Tensor-first contract.** The current `PoseInferenceBackend.predict_batch(
Sequence[np.ndarray]) -> List[PoseResult]` (`pose/types.py:56`) *is* a CPU
roundtrip by construction — which is why `predict_batch_cuda` exists as an
off-Protocol escape hatch probed by `hasattr` at `stages/pose.py:301` and
implemented only by SLEAP (`sleap.py:1056`). The new contract makes the
device-resident tensor path **primary**, with the numpy `predict_batch` as a thin
compatibility shim over it.

**SLEAP migration: characterization tests first.** `tools/equivalence/` already
provides this — `verify_sleap_exported_vs_service.py` constructs backends through
the production selector and compares keypoints pixel-for-pixel on real crops. Pin
behavior there **before** touching `sleap.py`; migrate only once parity is
provable. See also `tools/equivalence/PARITY_AUDIT.md`.

### Spec 3 — Config reshape, registry, and runtime integration
**Status:** deferred

`PoseRuntimeConfig` (`pose/types.py:22-44`) is structurally non-generic — flat
`yolo_*` / `sleap_*` fields and `backend_family: str  # yolo | sleap`. Reshape to
per-backend sub-dicts so backends own a config namespace. Replace the if/elif in
`create_pose_backend_from_config` (`pose/api.py:32`, raising at `:159`) with a
dict registry.

**Do not build entry-point plugin discovery.** That is N=1 speculation; the right
shape is learned from the second and third native backend.

Known hardcoded `"yolo" | "sleap"` sites to widen: `pose/types.py:26`;
`pose/api.py:33,53,99,159`; `pose/utils.py:350-368`;
`core/inference/config.py:75,193,350`; `core/inference/api.py:60`;
`core/inference/stages/pose.py:83,143`; `core/tracking/worker.py:4605,4644-4646`;
`core/identity/properties/cache.py:171,194` (cache identity!);
`runtime/resolver.py:15`; `runtime/compute_runtime.py:187,204,305`;
`utils/gpu_utils.py:380-415`; `posekit/gui/main_window.py:4225,4258,4359,4674,4762,4921`.

Follow the 9-step checklist in `docs/developer-guide/runtime-integration.md`.

### Spec 4 — Fine-tuning
**Status:** deferred

**Decided 2026-07-16:** native torch training script **launched as a subprocess
from PoseKit**. This preserves the repo convention that training shells out (cf.
`build_ultralytics_command`, `training/runner.py:111`; SLEAP trains in a separate
conda env) while still giving us layer-wise LR decay and final-layer surgery,
which are exactly what shelling out to a third-party CLI makes awkward.

Data source is already built: `build_coco_keypoints_dataset`
(`posekit/core/extensions.py:1138`) emits COCO keypoints, ViTPose's native format.

Recipe notes for when we get here — AdamW `lr=5e-4` (1e-3 for ViTPose+),
`wd=0.1`, grad clip 1.0, `JointsMSELoss(use_target_weight=True)`, Gaussian targets
`sigma=2` UDP-encoded. Layer-wise LR decay arithmetic: `num_layers = depth + 2`;
`pos_embed`/`patch_embed` → id 0, `blocks.{i}` → `i+1`, head → `num_layers-1`;
`lr_scale = decay ** (num_layers - layer_id - 1)`. The `+2` is easy to get wrong
and materially changes backbone LR.

For a new dataset: re-init `final_layer` only, load `strict=False` and **assert
the only missing/unexpected keys are `final_layer.*`**; fix `flip_pairs` for the
skeleton (or disable flip entirely if no L/R symmetry); disable half-body
transform (COCO-anatomy-specific); cut drop_path (~0.1, not 0.3) and epochs
(~20-50, not 210) for small data.

## Why this ordering

**Every genuine unknown lives in Spec 1.** Does `padding=2` matter as much as the
source says? Does an on-device UDP decode match cv2? Do the MoE `fc2` shapes load?
Those are answerable in a standalone module against an objective oracle (COCO AP)
that does not care about this repo.

**Specs 2 and 3 are engineering with known outcomes.** They are work, but nothing
about them should surprise us — and they are the ones that touch working code.

Doing the refactors first would mean reshaping config and migrating SLEAP onto a
runtime layer whose requirements are *guessed* from a ViTPose that has not run
yet. If the port then reveals that (say) the on-device decode belongs in the
runtime contract rather than the backend, we reshape twice. Spec 1 first means
Spec 2's abstraction is designed against two backends that both demonstrably work
— the same argument that rules out a premature plugin registry, applied to
ourselves.

## Environment fix: duplicate OpenMP runtime (applied 2026-07-16)

**Symptom:** `import torch, cv2` aborted the process (`OMP: Error #15`, exit 134).
`import cv2, torch` was fine. This bit any test importing both — i.e. all of
Spec 1.

**Cause:** two libomp copies mapped, and their *install names* diverged so dyld
could not dedupe them:

| copy | install_name |
|---|---|
| `$CONDA_PREFIX/lib/libomp.dylib` (llvm-openmp 22.1.2) | `@rpath/libomp.dylib` |
| `.../site-packages/torch/lib/libomp.dylib` (vendored) | `/opt/llvm-openmp/lib/libomp.dylib` |

torch links `@rpath/libomp.dylib` and resolves it via its own rpath to the
vendored copy, which dyld then registers under a different name than conda's.

**Fix applied** (to the `hydra-mps` env):

```bash
E=$CONDA_PREFIX  # hydra-mps
T=$E/lib/python3.13/site-packages/torch/lib/libomp.dylib
cp -p "$T" "$T.orig-backup"
ln -sf "$E/lib/libomp.dylib" "$T"
```

Safe because the LC_ID_DYLIB compat versions match exactly (current 5.0.0,
compatibility 5.0.0). Verified after: exactly one libomp maps; all import orders
work; torch matmul + MPS compute fine; pre-existing pose suite still 29 passed;
SLEAP/YOLO backends import.

**⚠️ Not persistent.** `make env-create-mps` (or any torch reinstall) restores the
vendored copy and the abort returns. Re-apply then.

**Two rejected alternatives**, recorded so they are not re-litigated:
- *`KMP_DUPLICATE_LIB_OK=TRUE`* — the repo's incumbent workaround
  (`trackerkit/app.py:18`, `tools/equivalence/runner.py:31`,
  `PARITY_AUDIT.md:83`). LLVM documents it as possibly producing *silently
  incorrect results*, which is the exact failure mode a numerical-parity port
  cannot tolerate. Now unnecessary; the `app.py` call is redundant but harmless.
- *"import cv2 before torch"* — **does not work.** isort sorts third-party
  `import torch` ahead of first-party `from hydra_suite...`, so torch wins the
  race in any module importing both. This was tried and removed.

**Root cause note:** OpenCV is also installed twice (conda-forge `opencv 4.13.0`
*and* pip `opencv-python`/`-headless 4.13.0.92`; the pip wheel currently shadows
conda's). That did not cause this abort — the pip wheel bundles no libomp — but
it is worth cleaning up separately.

## Export/artifact conventions in this repo (mapped 2026-07-16)

Read before adding export or runtime code to ANY backend. This is what the existing code
actually does — including where it disagrees with itself.

### The key structural fact

`auto_export_yolo_model` (`yolo.py:38`) and `auto_export_sleap_model` (`sleap.py:1353`) are
**not the export**. They are the caching wrapper — signature, location, sidecar, staleness —
and they delegate the actual conversion to ultralytics' `model.export()` and SLEAP's
conda-env exporter respectively. For ViTPose nobody supplies that conversion, so we own it,
and the layering falls out cleanly:

| layer | YOLO | SLEAP | ViTPose |
|---|---|---|---|
| recipe (torch -> artifact) | ultralytics `model.export()` | SLEAP exporter | `vitpose/export.py` (Spec 1, leaf) |
| caching wrapper | `auto_export_yolo_model` | `auto_export_sleap_model` | `backends/vitpose.py` (Spec 3) |
| lazy trigger | `api.py:75` | `api.py:117` | `api.py` (Spec 3) |

### There is no single canonical export module — there are four systems

| domain | export entry | file |
|---|---|---|
| YOLO pose | `auto_export_yolo_model` | `pose/backends/yolo.py:38` |
| SLEAP pose | `auto_export_sleap_model` | `pose/backends/sleap.py:1353` |
| OBB detector (legacy mixin) | `RuntimeArtifactMixin._prepare_runtime_artifact_for_task` | `core/detectors/_runtime_artifacts.py:1007` |
| OBB detector (clean rewrite) | `load_obb_executor` / `_export_artifact` | `core/inference/runtime_artifacts.py:412,123` |

`core/inference/runtime_artifacts.py` is an explicit, documented **clean rewrite** of the
legacy mixin, created to fix parity-audit finding **H4** (the legacy code silently fell back
to PyTorch instead of erroring). Its stance is the more recent and more defensible one.

### Conventions that are consistent everywhere — safe to copy

- **Co-locate the artifact with the source checkpoint.** `paths.get_models_dir()` exists but
  is used by ZERO export paths (0 hits in `core/`). Do not "improve" this.
- **Sidecar `<artifact>.runtime_meta.json`** holding the signature; written only after a
  successful export, checked before reuse.
- **Lazy, on-first-use export**, never a separate pre-export step; always on a `QThread`
  worker (`posekit/gui/workers.py` `QObject.run()`), never the GUI thread.
- **`pose/artifacts.py` is the shared primitive set** for the pose package —
  `path_fingerprint_token` (:71), `artifact_meta_path` (:82), `artifact_meta_matches` (:92),
  `write_artifact_meta` (:107). Both `yolo.py` and `sleap.py` use them. A ViTPose backend
  must too. (The detector code ignores them and reimplements its own — twice.)
- **`derive_onnx_execution_providers(compute_runtime)`** (`compute_runtime.py:160-184`) is
  the ONE genuinely canonical shared piece. Any ONNX session must get its providers there.
- **Python API export, never a `trtexec` subprocess** (zero hits repo-wide).

### Where the code contradicts itself — decide deliberately, do not copy blindly

1. **FP16.** OBB hardcodes `half=True` (`_runtime_artifacts.py:896`,
   `runtime_artifacts.py:162`). SLEAP deliberately keeps FP32 — `sleap.py:420-421`: "fp16 is
   deferred to preserve keypoint precision" — and `compute_runtime.py:141-142` states the
   same rule for the ORT-TRT-EP path. **DECIDED for ViTPose: FP32**, following the SLEAP
   keypoint precedent. A keypoint model with sub-pixel decoding is the SLEAP case, not OBB's.
2. **Failure semantics.** The pose package silently downgrades a failed export to
   native/service (`api.py:78-85`, `133-141`). The clean OBB rewrite raises
   `ArtifactExportError` precisely because silent fallback was a real parity bug (H4).
   **Unresolved — Spec 3 must choose.** The rewrite's stance looks right, but note it still
   defaults `auto_export=True`; it only refuses to silently downgrade to a *different runtime*.
3. **Signature scheme.** Hash-with-version-tag (pose package; OBB legacy bakes in
   `onnx_v4_static_imgsz{N}_opset17...` so a recipe change forces a rebuild) vs. plain
   mtime+imgsz freshness (clean rewrite, which traded that robustness for simplicity and
   LOST recipe-versioning). The pose package's own signatures have no version tag either —
   bumping `opset` in `yolo.py:93` would silently reuse stale artifacts. **A ViTPose signature
   should include a recipe version tag.**
4. **GPU fingerprinting in the TRT cache key: absent everywhere**, despite TensorRT engines
   being GPU/driver-specific. `_get_tensorrt_build_context()` collects `gpu_name`/`cuda_version`
   and writes them to the sidecar, but `_artifact_signature()` never includes them — so a
   `.engine` built on a different GPU is happily reused. Mismatches are caught only
   reactively, by sniffing exception text (`_is_fatal_tensorrt_environment_error`). **Copying
   "the convention" here copies a known gap.** Fixing it for ViTPose would be an improvement,
   not a departure.

### CoreML for pose does not exist

The pose runtime vocabulary (`normalize_runtime_flavor`, `pose/utils.py:350`) is
`native | onnx | tensorrt` — no CoreML. Only the OBB/detector world has CoreML, via
ultralytics' own `model.export(format="coreml")` (`runtime_artifacts.py:168-176`), which we
cannot use for a hand-rolled model. `YoloNativeBackend`'s "CoreML" is ORT's
`CoreMLExecutionProvider` at *inference* time, not a `.mlpackage` export.

Commit `ebf5296` ("Apple GPU-Fast to CoreML with native-MPS fallback") is a **resolver**
change, not an export mechanism: it makes the Apple `gpu_fast` tier return CoreML only if an
artifact already exists, else fall back to native MPS.

So **CoreML for ViTPose is new design, not convention-following.** Risk: `coremltools 9.0`
warns torch 2.11 is untested (max tested 2.7).

### Verification convention

`tools/equivalence/verify_sleap_exported_vs_service.py` is the template: build BOTH backends
through the *production selector* (`create_pose_backend_from_config`), feed identical real
crops, compare keypoints pixel-for-pixel. `PARITY_AUDIT.md` is the living ledger of
findings. A ViTPose backend should get an analogous `verify_vitpose_*.py` at Spec 3.

### Doc drift to be aware of

`docs/developer-guide/runtime-integration.md:21` cites
`src/hydra_suite/core/runtime/compute_runtime.py`; the real path has no `core/`. Line 29
references `derive_detection_runtime_settings(...)`, which has been **deleted**. The
checklist is structurally authoritative; its function/path names are not.

## Conventions that constrain all four specs

- **Dependency direction** (CLAUDE.md:159-165, 192-198): Core/Runtime/Data/
  Training/Utils never import from app layers or Integrations. The backend belongs
  at `core/identity/pose/backends/vitpose.py`.
- **~500-line rule** (CLAUDE.md:123). `yolo.py` (293 lines) is the model to follow;
  `sleap.py` (1780) is the cautionary tale.
- **Paths** (CLAUDE.md:199-201): never `Path(__file__).parents[N]`; use
  `hydra_suite.paths` (`get_models_dir()`, `get_skeleton_dir()`).
- **Canonical runtimes** (CLAUDE.md:257): `cpu`, `mps`, `cuda`, `onnx_cpu`,
  `onnx_cuda`, `tensorrt`.
- **torch is deliberately unpinned** and absent from `pyproject.toml` (needs a
  custom index URL; see `requirements.txt:5-7`).
- **Pre-PR** (CLAUDE.md:74-79): `make commit-prep` → `make lint-moderate` →
  `make docs-check`. Line length 88.

## Open questions / unverified

- OneDrive checkpoint liveness for non-plus L/H (403 to `curl`; needs a browser).
- Whether HF's decode `sigma=0.8` deviation is deliberate — no comment or issue
  found. See Spec 1 design for why it matters.
- ViTPose+ effective batch size: config says `samples_per_gpu=64`; GPU count is
  not in the config, so the 512 effective figure is inferred.
