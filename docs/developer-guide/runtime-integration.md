# Runtime Integration Guide

This guide defines the runtime contract for end-to-end integration of:

- New detection models
- New pose models
- New identity/individual-analysis methods (classifiers, embeddings, contrastive features, tag readers)

## Design Goal

All compute-heavy methods must be controlled by one canonical runtime setting:

- `compute_runtime`

No feature should require users to configure a separate runtime selector.

## Source of Truth

Runtime support and translation logic are centralized in:

- `src/hydra_suite/core/runtime/compute_runtime.py`
- `src/hydra_suite/utils/gpu_utils.py`

Core public helpers:

- `CANONICAL_RUNTIMES`
- `allowed_runtimes_for_pipelines(...)`
- `infer_compute_runtime_from_legacy(...)`
- `derive_detection_runtime_settings(...)`
- `derive_pose_runtime_settings(...)`

## Canonical Runtime Values

- `cpu`
- `mps`
- `cuda`
- `rocm`
- `onnx_coreml`
- `onnx_cpu`
- `onnx_cuda`
- `onnx_rocm`
- `tensorrt`

## Integration Checklist (Required)

### 1) Define a pipeline key

Add a stable pipeline name and use it in runtime gating.

Current examples:

- `yolo_obb_detection`
- `yolo_pose`
- `sleap_pose`
- `cnn_identity`
- `head_tail`

For future additions, use names like:

- `appearance_embedding`
- `contrastive_embedding`
- `apriltag_classifier`
- `colortag_classifier`

### 2) Add capability rules

Update `_pipeline_supports_runtime(...)` in `compute_runtime.py` so the new pipeline explicitly defines supported runtimes.

Rules must be strict:

- If unsupported, return `False`.
- Do not silently remap unsupported runtime to a different backend.

### 3) Add runtime translation

If the pipeline consumes legacy backend knobs, add mapping from `compute_runtime` to backend settings.

Examples already used:

- Detection: `yolo_device`, `enable_onnx_runtime`, `enable_tensorrt`
- Pose: `pose_runtime_flavor`, `pose_sleap_device`

### 4) Wire UI intersection gating

Ensure the UI includes the new pipeline in the runtime context set.

TrackerKit pattern:

- Gather enabled pipeline set.
- Call `allowed_runtimes_for_pipelines(...)`.
- Populate runtime dropdown from the intersection.

PoseKit uses the same pattern for active prediction backend scope.

### 5) Implement runtime lifecycle

If the integration has long-lived resources (service/subprocess/session), lifecycle must be run-scoped:

- Initialize once per run.
- Warmup once.
- Close on complete/error/cancel.

Use existing runtime manager/service patterns where possible.

### 6) Export artifacts automatically

If ONNX/TensorRT export is needed:

- Generate artifacts automatically.
- Store artifacts adjacent to model paths.
- Save runtime metadata signature for freshness checks.
- Never require a manual export path for normal operation.

### 7) Keep cache keys runtime-correct

Any cached output that depends on runtime/model/export shape must include those inputs in cache identity.

For new features:

- Include model fingerprint and runtime flavor in cache signatures.
- Include feature-specific shaping params (for example max instances, embedding dimension, preprocessing mode).

### 8) Lock controls during compute

UI controls that could invalidate active runtime sessions must be disabled while jobs are running.

This prevents mid-run backend switches and thread crashes.

### 9) Add tests (minimum bar)

Add/extend tests for:

- Capability matrix and intersection gating.
- Runtime translation determinism from `compute_runtime`.
- Migration from legacy config values.
- Lifecycle correctness (startup/teardown on success and failure).
- Artifact auto-export + freshness behavior.
- Failure fallback behavior with explicit logging.

## End-to-End Acceptance Criteria

A new model/method integration is complete only when:

1. It appears in runtime gating with explicit support rules.
2. It runs from canonical `compute_runtime` without extra runtime selectors.
3. Its ONNX/TensorRT artifacts are auto-managed (if applicable).
4. Caches remain valid and runtime/model-aware.
5. TrackerKit and PoseKit behavior is consistent where the feature exists.

## Common Anti-Patterns (Do Not Add)

- Hidden runtime remapping (`onnx_*` requested, CPU used without notice).
- Feature-specific runtime dropdowns when global runtime is available.
- Manual exported-model-path requirements for standard workflows.
- Runtime checks scattered across GUI/business logic without shared resolver usage.

## Related Docs

- [GPU Backends](gpu-backends.md)
- [Extending Detection](extending-detection.md)
- [Extending Identity](extending-identity.md)
- [Compute Runtimes (User Guide)](../user-guide/compute-runtimes.md)
