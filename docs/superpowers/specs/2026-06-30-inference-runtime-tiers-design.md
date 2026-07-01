# Inference Runtime Tiers — Unified Design

**Status:** Approved design (north star). Implemented in phases; each phase gets
its own implementation plan derived from this document.

**Date:** 2026-06-30

**Supersedes for runtime selection:** the 7-runtime `compute_runtime` model
(`cpu`, `mps`, `cuda`, `onnx_cpu`, `onnx_cuda`, `onnx_coreml`, `tensorrt`) and
per-stage runtime selection.

---

## 1. Motivation

The inference pipeline currently exposes **seven** compute runtimes, selectable
**per stage** (OBB, head-tail, CNN, pose each carry their own
`compute_runtime`). Benchmarking (`tools/equivalence/opt_microbench.py`, CUDA
RTX 6000 Ada + Apple M3 Max, 2026-06-30) showed most of these options are
dominated or broken:

- `onnx_cuda` classifier: **12% slower** than native `cuda` — strictly dominated.
- `onnx_cpu` for OBB: **broken on non-CUDA hosts** — the direct OBB executor
  hardcodes `cuda:0` staging and requires `CUDAExecutionProvider`
  (`_direct_obb_runtime.py:76,402-410`).
- `onnx_coreml` for OBB: fails on dynamic shapes (E5RT unbounded dimension).
- `tensorrt` classifier: only **4.5% faster** than native `cuda`, and non-exact.
- `onnx_coreml` **classifier** on Mac: **1.4–1.7× faster** than native MPS — a
  genuine win, the one ONNX case worth keeping.

Per-stage selection also permits inefficient mixing (e.g. `cuda` OBB +
`onnx_cuda` classifier + `tensorrt` pose) that forces avoidable data movement.

Exact (determinism-preserving) GPU optimizations have narrow headroom: `channels_last`
gives **+11%** on CUDA CNN compute but is **−54% (harmful) on MPS**;
`inference_mode` ≈ `no_grad` (already present); pinned/`non_blocking` H2D is
+14% on a copy that is tiny relative to compute. The large (2–5×) speedups live
only in non-exact fp16 paths (TensorRT / CoreML).

## 2. North-Star Principle

**One runtime tier for the entire pipeline, auto-resolved to the best backend
per platform.** Native tiers are exact and device-invariant; the fast tier
trades bit-exactness for speed. No tensor data crosses devices within a run.

## 3. The Three User-Facing Tiers

| Tier | CUDA host resolves to | Apple host resolves to | CPU-only host | Numerics contract |
|---|---|---|---|---|
| **CPU** | torch CPU | torch CPU | torch CPU | Exact / bit-reproducible |
| **GPU** | torch CUDA + exact wins (`channels_last`, pinned H2D, `inference_mode`) | torch MPS | falls back to CPU tier | Exact; device-invariant within existing ~0.006 px FP-noise envelope |
| **GPU-Fast** | TensorRT fp16 engines per stage; native-CUDA fallback per stage | CoreML `.mlpackage` (YOLO) + CoreML classifier path; native-MPS fallback | falls back to CPU tier | **Not** bit-identical to native; **deterministic run-to-run**; labeled "may reduce accuracy" |

**Rules:**

- **Pipeline-wide, single tier.** One `InferenceConfig.runtime_tier` value
  applies to every stage. No per-stage runtime field exists.
- **Best-effort fast-mode.** In `GPU-Fast`, any stage lacking a fast artifact
  (export unsupported, export failed, or no benefit) runs **native-GPU on the
  same device** — never CPU. Logged, never a hard crash.
- **AprilTag exception.** AprilTag always runs on CPU (a C-library detector, not
  a GPU model). It is exempt from the tier and is the sole stage that reads a
  CPU-side frame; this is inherent to AprilTag and independent of the tier.

## 4. Central Component: `RuntimeResolver`

A new component in the `runtime/` layer is the single authority mapping
**(tier, platform, stage, artifact availability) → (backend, device)**. It
absorbs logic currently scattered across `_pipeline_supports_runtime`,
`allowed_runtimes_for_pipelines`, `derive_onnx_execution_providers`, and the
per-stage `compute_runtime` fields.

- Stages ask the resolver what to load; stages no longer carry runtime strings.
- The resolver is pure/deterministic and unit-testable in isolation: given a
  tier, a detected platform, a stage key, and a callable reporting artifact
  availability, it returns the concrete backend + device (or the native-GPU
  fallback).
- Platform detection reuses `hydra_suite.utils.gpu_utils` / existing
  availability flags.

**Interface sketch** (names finalized during planning):

```python
RuntimeTier = Literal["cpu", "gpu", "gpu_fast"]

@dataclass(frozen=True)
class ResolvedBackend:
    backend: Literal["torch", "tensorrt", "coreml"]
    device: Literal["cpu", "cuda", "mps"]
    used_fallback: bool          # True if fast-mode fell back to native-GPU

class RuntimeResolver:
    def __init__(self, tier: RuntimeTier, platform: PlatformInfo): ...
    def resolve(self, stage: str, artifact_available: Callable[[], bool]) -> ResolvedBackend: ...
```

## 5. User-Facing Changes (GUI) — "UI matches the backend"

The GUI is a first-class deliverable: whenever the backend runtime model
changes, the UI changes in the same phase so the two never drift. No phase that
alters user-facing runtimes may merge without its UI update.

**5.1 Remove the old options.** `onnx_cpu`, `onnx_cuda`, `onnx_coreml`,
`tensorrt` are removed as user-facing runtimes. CoreML, ORT, and TensorRT
survive only as **internal fast-mode backends** the resolver selects.

**5.2 Collapse to one tier selector.** The three trackerkit dropdowns
(`combo_compute_runtime`, `combo_headtail_runtime`, `combo_cnn_runtime` in
`trackerkit/gui/panels/setup_panel.py`) and the posekit `combo_pred_runtime`
(`posekit/gui/main_window.py`) are replaced by **one tier selector**, bound to
`InferenceConfig.runtime_tier` (and the kit config schema). The old per-stage
combos and their signal wiring are deleted.

**5.3 Platform-aware labels.** The selector shows only what the host supports,
labeled by the backend the resolver will actually use:

- CUDA host: `CPU` / `GPU (CUDA)` / `GPU-Fast (TensorRT)`
- Apple host: `CPU` / `GPU (Metal)` / `GPU-Fast (CoreML)`
- CPU-only host: `CPU` only (GPU tiers hidden/disabled)

Labels are derived from the same `RuntimeResolver`/platform detection as the
backend, so the dropdown text is always truthful about the selected backend.

**5.4 Reflect resolution + fallback.** When `GPU-Fast` is selected, the panel
surfaces the resolver outcome — a concise, read-only indicator (e.g. status
line / tooltip) noting when a stage fell back to native-GPU (`used_fallback`)
so the user understands what actually ran. Non-blocking; informational.

**5.5 No stale state.** `allowed_runtimes_for_pipelines`-driven population is
replaced by resolver-driven tier enumeration; removing a runtime from the
backend removes it from every kit's UI in the same change.

## 6. Migration (Hard Cutover)

Per-stage `compute_runtime` fields are replaced by a single pipeline-level
`runtime_tier`. On loading any legacy config, values are mapped:

| Legacy per-stage value | New tier |
|---|---|
| `cpu` | `CPU` |
| `cuda`, `mps` | `GPU` |
| `onnx_cpu`, `onnx_cuda`, `onnx_coreml`, `tensorrt` | `GPU-Fast` |

If a legacy config mixed values across stages, the highest tier present wins
(`GPU-Fast` > `GPU` > `CPU`), and a one-line warning is logged. Per-stage fields
are dropped after mapping. Applies both to the new `InferenceConfig`
(`from_json`) and to trackerkit's on-`main` config format via its translation
layer into `InferenceConfig`.

## 7. Fast-Mode Export Coverage (hard requirement)

Every model family the pipeline can run must have a fast-mode path, or a
logged native-GPU fallback. Coverage matrix:

| Family | arch id(s) | ONNX exporter (→ TensorRT) | CoreML exporter |
|---|---|---|---|
| YOLO detect/pose/cls | `yolo`, `yolo_multihead`, `classifier_multihead` | exists: ultralytics `.export(format="onnx")` | exists: ultralytics `.export(format="coreml")` |
| Tiny classifier | `tinyclassifier` | exists: `export_tiny_to_onnx` | **new (P3):** `export_tiny_to_coreml` |
| torchvision + timm classifier | `resnet50`, `convnext_tiny`, `efficientnet_b0`, … | exists: `export_torchvision_to_onnx` | **new (P3):** `export_torchvision_to_coreml` |
| SLEAP pose | (SLEAP) | exists (SLEAP export) | native-GPU fallback (out of scope for new CoreML) |
| AprilTag | — | N/A (CPU-only) | N/A |

- Multihead classifiers export **per-factor backend**.
- TensorRT engines are built from the ONNX peer (all three classifier families
  already have ONNX exporters), so P2 covers every family on CUDA.
- CoreML exporters for tiny + torchvision/timm are the new P3 deliverable; until
  then, Apple `GPU-Fast` falls back to native-MPS for those families
  (best-effort rule, §3).

## 8. Phasing

Each phase is a separate implementation plan; all share this document.

### Phase 1 — Native-GPU exact wins (independent, ships first)
- `channels_last` **gated on the actual torch device == cuda** (never applied on
  MPS, where it regresses 54%). Applies to the conv-heavy classifier and
  YOLO/OBB torch paths.
- `inference_mode` around GPU crop extraction (`canonicalization/crop.py`
  `grid_sample`, lines 299/387); upgrade classifier `no_grad` → `inference_mode`.
- Pinned + `non_blocking=True` H2D for crop uploads (CUDA only).
- No taxonomy change: gated on device, not tier — lands immediately.
- Expected: ~11% CNN compute on CUDA; MPS/CPU unchanged; numerics within the
  existing device-invariance envelope (documented caveat: `channels_last` is not
  bit-identical to the prior CUDA output but stays within ~0.006 px FP noise).

### Phase 2 — Runtime tier taxonomy
- Introduce `RuntimeResolver` + `InferenceConfig.runtime_tier`.
- Single tier pipeline-wide; remove per-stage `compute_runtime`.
- TensorRT becomes the CUDA `GPU-Fast` backend (wire existing auto-export in
  `runtime_artifacts.py`); best-effort native-CUDA fallback per stage.
- Remove ONNX/TensorRT as user-facing runtimes; collapse GUI to one tier
  selector (§5.1–5.5) in **both** trackerkit and posekit, delete the old
  per-stage combos + wiring, and update each kit's config schema
  (`<kit>/config/schemas.py`) to carry `runtime_tier`. **The UI change ships in
  this phase, not later** — backend and UI move together.
- Hard-cutover migration (§6).
- Phase 1 exact wins fold under the resolved `GPU` tier.

### Phase 3 — Native CoreML fast-mode (Apple)
- ultralytics `.mlpackage` export for YOLO OBB/pose.
- New `export_tiny_to_coreml` and `export_torchvision_to_coreml` (covers timm).
- CoreML artifact auto-management mirroring the TensorRT pattern.
- Wire CoreML backends into `GPU-Fast` on Apple; best-effort native-MPS fallback.
- **UI:** the `GPU-Fast (CoreML)` label (§5.3) becomes live on Apple hosts once
  the CoreML backend is wired; the fallback indicator (§5.4) reflects any
  family that still uses native-MPS.

## 9. Testing

- **Resolver:** unit tests for every (tier × platform × artifact-availability)
  combination → expected `ResolvedBackend`, including fallback paths.
- **Migration:** mapping tests for each legacy value and the mixed-config
  highest-tier rule.
- **Exactness (P1/P2):** CPU and GPU tiers preserve existing equivalence +
  device-invariance guarantees (reuse `tools/equivalence`).
- **Fast-mode determinism (P2/P3):** same input → identical output across two
  runs (deterministic, though not bit-identical to native).
- **Export coverage (P2/P3):** each classifier family (yolo/tiny/torchvision+timm,
  including multihead per-factor) exports and runs under the fast tier, or logs
  a native-GPU fallback.
- **GUI:** tier selector shows correct platform-specific labels (§5.3), hides
  unavailable tiers, binds to `runtime_tier` in the kit config schema, and the
  old per-stage combos are gone from both trackerkit and posekit; the
  resolution/fallback indicator (§5.4) renders when `GPU-Fast` is selected.

## 10. Error Handling

- Fast-mode export/build failure → logged warning + native-GPU fallback on the
  same device (never CPU, never crash).
- Requesting a GPU tier on a CPU-only host → resolves to CPU tier with a logged
  notice (GUI hides the option, but config-loaded/headless runs degrade
  gracefully).
- Migration of an unrecognized legacy runtime string → maps to `GPU` if it is in
  the CUDA/MPS family else `CPU`, with a warning.

## 11. Non-Goals

- TF32 and cudnn.benchmark are **not** enabled (they break determinism without
  belonging to the fast-mode export model); may be revisited as a separate
  fast-mode sub-option later.
- `onnx_rocm` and generic ONNX portability are out of scope; ONNX remains an
  internal IR for TensorRT engine builds only.
- SLEAP internal runtime handling is unchanged beyond tier mapping.
