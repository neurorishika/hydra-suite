# TrackerKit Streaming Individual Analysis Plan

Date: 2026-04-25

Related docs:

- `runtime-integration.md`
- `tracking-algorithm-deep-dive.md`
- `trackerkit-identity-overhaul-spec.md`

## Objective

Move TrackerKit individual analysis for pose and CNN identity from the current replay/precompute-heavy workflow to a streaming-first workflow that runs immediately after:

1. YOLO OBB detection
2. detection filtering
3. head-tail prediction
4. head-tail-based canonical reorientation

The target execution order for filtered detections is:

`detect -> filter -> head-tail -> canonical reorientation -> CNN + pose`

AprilTag remains separate and continues to run on AABB crops. It is explicitly out of scope for this migration except where orchestration must keep it isolated.

This plan now serves as the transport and runtime substrate for the identity overhaul described in `trackerkit-identity-overhaul-spec.md`. The streaming work owns the payload, crop transport, runtime, and cache-emission path; the identity overhaul owns the probabilistic evidence model and online and offline decoding built on top of that substrate.

## Why This Change

The current default path pays avoidable costs:

- A second full video/precompute pass is forced for non-realtime YOLO OBB runs when pose or CNN analysis is enabled.
- Unified precompute re-extracts crops from CPU frames even when head-tail has already derived canonical orientation metadata.
- CNN precompute currently uses CPU-style crop batches rather than a GPU-native crop path.
- On CUDA, this leaves performance on the table by forcing extra CPU crop extraction and host/device transfers.
- On MPS, it still incurs the cost of replaying the video even though TensorRT/ONNX CUDA optimizations are not relevant there.

The fastest path to meaningful speedup is not full removal of every fallback path. It is to make streaming filtered-detection analysis the primary execution path for pose and CNN, then progressively demote the replay path to a cache/recovery mechanism.

## Scope

### In Scope

- Run CNN and pose immediately after head-tail reorientation for filtered detections.
- Reuse detector-stage canonical orientation data instead of recomputing orientation downstream.
- Make the CNN path GPU-native-ready in the same way head-tail already is.
- Support all relevant runtimes for CNN and pose:
  - native CPU
  - native MPS
  - native CUDA
  - ONNX CPU
  - ONNX CUDA
  - TensorRT
- Preserve cache artifacts required by downstream tracking/orientation/export logic.
- Keep replay/offline cache generation functional during migration.

### Out of Scope

- AprilTag redesign beyond keeping it separate.
- Full removal of interpolated occlusion post-pass.
- Full deletion of UnifiedPrecompute in the first implementation wave.
- Any CLI or user-visible workflow changes unrelated to execution order and performance.

## Current State Summary

### What Exists Today

- Head-tail already executes in the detector path and can use GPU-native crop inference on CUDA-capable inputs.
- Detector output can persist canonical affine metadata alongside raw detections.
- The worker already has a streaming/live precompute mode for advanced individual analysis.
- PosePipeline already has support for CUDA crop batches when given CUDA tensors.
- The lower classifier backend already has a GPU-native batch entry point.

### What Still Blocks The Desired Path

- Unified precompute remains the default for non-realtime pose/CNN analysis.
- Unified precompute reopens/replays the video and extracts crops from CPU frames.
- CNNPrecomputePhase uses CPU crop batches and does not expose a higher-level GPU-native API parallel to pose/head-tail.
- The current live path still routes through the generic precompute crop extraction layer instead of chaining directly off detector-stage filtered detections and their canonical orientation products.
- Runtime support is not expressed through a single shared crop-batch abstraction across head-tail, pose, and CNN.

## Architectural Direction

### Primary Principle

The filtered-detection boundary becomes the handoff point for downstream individual analysis.

Everything after detection filtering should consume one shared per-frame/per-batch payload that contains:

- filtered detection ids
- filtered OBB corners
- resolved canonical orientation / reorientation metadata
- canonical affine matrices
- optionally canonical crops as CPU arrays
- optionally canonical crops as GPU tensors
- source runtime metadata needed for downstream backend routing

### Resulting Execution Model

#### Primary Path

- Forward YOLO OBB run
- Filter detections once
- Run head-tail once on filtered detections
- Produce canonical orientation metadata once
- Dispatch the same filtered detections and oriented crop payload to:
  - pose analysis
  - CNN analysis
- Write pose/CNN outputs directly into live stores and cache writers
- Consume those outputs during the same tracking run

#### Fallback Path

- Replay from detection cache when live streaming is unavailable, disabled, or explicitly requested
- Rebuild pose/CNN artifacts offline from cached detections without needing fresh detector inference
- Keep AprilTag independent

### Explicit Non-Goal For Early Phases

Do not make pose or CNN depend on AprilTag crop format or AABB crop rules. Pose and CNN will standardize around canonical/oriented crops; AprilTag keeps its AABB-only path.

## Dependency On The Identity Overhaul

The identity overhaul depends on this plan for three concrete reasons:

- the online identity decoder must consume live filtered-detection outputs without forcing a replay path
- the posterior evidence cache must be emitted from the same live and replay execution path
- the final offline identity decoder must consume artifacts that are stable regardless of whether the run used live streaming or replay fallback

Because of that dependency, this plan should be executed first wherever it establishes shared infrastructure. The identity overhaul can begin in parallel only once the shared payload and cache-emission points are defined.

## Runtime Strategy

### Common Runtime Expectations

Head-tail should become the reference model for how pose and CNN are integrated into the forward loop.

#### Native CPU

- Supported for correctness and fallback.
- Uses CPU crop arrays and CPU backend execution.

#### Native MPS

- Supported primarily through native backend execution.
- Should benefit from removal of replay/video re-read even if crop handling remains CPU-oriented in early phases.
- No dependency on ONNX CUDA or TensorRT.

#### Native CUDA

- Primary optimization target.
- Prefer detector-stage GPU-resident canonical crops or a GPU-native crop extraction path immediately after head-tail.
- Avoid GPU -> CPU -> GPU round-trips where backend supports device-native execution.

#### ONNX CPU

- Supported as an offline/fallback runtime.
- Can use CPU crop arrays.

#### ONNX CUDA

- Supported for pose and CNN where backend export/runtime already exists.
- Requires a shared GPU-native batch-preprocess path and backend-level I/O binding support where available.

#### TensorRT

- Supported for CUDA-capable systems only.
- Must use the same runtime artifact bookkeeping rules already used elsewhere in TrackerKit.

## Required Design Changes

### 1. Introduce A Shared Oriented Analysis Payload

Create a focused data contract for downstream filtered-detection analysis. This should not be a copy of raw detector output; it should be post-filter and post-head-tail.

Recommended fields:

- `frame_idx`
- `detection_ids`
- `obb_corners`
- `headtail_heading`
- `headtail_confidence`
- `headtail_directed`
- `canonical_affines`
- `canonical_crops_cpu` optional
- `canonical_crops_cuda` optional
- `input_is_bgr`
- `runtime_family`

This payload must also become the attachment point for identity evidence emission. It does not need to carry final posterior vectors itself, but it must carry enough stable indexing information for a downstream identity phase to emit per-detection evidence sidecars using the same `frame_idx`, detection ordering, and crop provenance in both live and replay modes.

This payload becomes the single handoff object from detection/head-tail to pose and CNN.

### 2. Add A Public GPU-Native CNN Batch API

The lower classifier backend already has CUDA batch support. The high-level CNN identity wrapper needs an explicit public method parallel to pose/head-tail.

Recommended API addition:

- `CNNIdentityBackend.predict_batch_cuda(crops_chw, input_is_bgr=True)`

Responsibilities:

- preserve current thresholding and class mapping semantics
- preserve factor/scoring mode behavior
- expose calibrated posterior output hooks so the identity overhaul can emit full-catalog evidence without duplicating inference
- route to CPU automatically when the configured execution backend cannot stay device-native
- expose profiling hooks consistent with pose/head-tail where practical

### 3. Split Reorientation From Replay Infrastructure

Canonical reorientation data currently survives mainly as affine metadata. Make that orientation output a first-class product of the detector/head-tail stage so downstream analysis can consume it directly without going back through generic precompute.

### 4. Promote Streaming Analysis To The Default Forward Path

For YOLO OBB forward runs with pose/CNN enabled:

- do not force unified precompute before tracking
- do not reopen the video solely for pose/CNN precompute
- do not require a second crop extraction path when filtered detections already produced everything needed

### 5. Keep Replay As An Explicit Fallback

Replay remains valuable for:

- recovery from interrupted runs
- cache regeneration from detection cache
- debug repro
- environments where live/runtime support is incomplete

Replay should become an explicit compatibility mode, not the primary execution path.

## Phased Implementation Plan

## Phase 0: Guardrails And Instrumentation

### Phase 0 Goal

Establish baseline metrics and lock down contracts that must not regress during the migration.

### Phase 0 Tasks

- Capture baseline timings using real-video probes, not synthetic benchmark dialog numbers.
- Document the current contract for batched raw detection results and detection cache metadata.
- Preserve the existing batched raw detector return contract: `detect_objects_batched(return_raw=True)` must continue returning the full raw tuple, including `canonical_affines`, on empty/failure paths so detection-cache writers do not break.
- Add focused logging/profiling counters for:
  - frame read time
  - filter time
  - head-tail time
  - canonical crop extraction time
  - pose enqueue/inference time
  - CNN enqueue/inference time
  - cache write time
- Confirm which downstream consumers require:
  - raw detection cache
  - canonical affines
  - detected properties cache
  - pose properties cache
  - CNN identity cache

### Phase 0 Files Expected

- `src/hydra_suite/core/tracking/worker.py`
- `src/hydra_suite/core/tracking/detection_phase.py`
- `src/hydra_suite/trackerkit/obb_runtime_probe.py`
- `tools/benchmark_trackerkit_obb.py`

### Phase 0 Exit Criteria

- A before/after measurement method is agreed and reproducible.
- Existing batched raw contract remains unchanged.
- There is no ambiguity about which caches are authoritative for each stage.

## Phase 1: Shared Streaming Analysis Contract

### Phase 1 Goal

Create a clean handoff object for filtered-detection, post-head-tail analysis.

### Phase 1 Tasks

- Add a small typed container for the streaming analysis payload.
- Build it immediately after filtered detections and head-tail results are available.
- Include canonical affine data from detector/head-tail output.
- Allow payloads to carry either CPU crops, CUDA crops, or both.
- Keep AprilTag out of this contract.
- Define stable detection-slot indexing that later identity evidence caches can reuse unchanged.

### Phase 1 Files Expected

- `src/hydra_suite/core/tracking/worker.py`
- `src/hydra_suite/core/detectors/yolo_detector.py`
- possibly a new module under `src/hydra_suite/core/tracking/`

### Phase 1 Exit Criteria

- Pose and CNN can both consume the same payload shape.
- No stage after head-tail needs to rediscover orientation rules.

## Phase 2: GPU-Native CNN API And Runtime Readiness

### Phase 2 Goal

Make CNN analysis symmetrical with head-tail and pose in terms of runtime-aware batch execution.

### Phase 2 Tasks

- Add high-level `predict_batch_cuda` to `CNNIdentityBackend`.
- Add a posterior-producing path or hook that can expose calibrated class probabilities for the identity overhaul.
- Ensure runtime-specific behavior is preserved for:
  - native CPU
  - native MPS
  - native CUDA
  - ONNX CPU
  - ONNX CUDA
  - TensorRT
- Centralize runtime/profile metadata exposure if missing.
- Add artifact/export handling consistent with runtime artifact policies.
- Validate scoring mode parity between CPU and CUDA entry points.

### Phase 2 Files Expected

- `src/hydra_suite/core/identity/classification/cnn.py`
- `src/hydra_suite/core/identity/classification/backend.py`
- `src/hydra_suite/core/detectors/_runtime_artifacts.py`
- any shared runtime config helpers used by CNN model resolution

### Phase 2 Exit Criteria

- CNN can consume CUDA crop batches directly when available.
- CPU and GPU entry points produce equivalent semantic outputs.
- ONNX/TensorRT runtime selection remains compatible with existing artifact bookkeeping.

## Phase 3: Streaming Pose Integration On The Filtered Path

### Phase 3 Goal

Feed pose from the same post-head-tail payload instead of replaying frames for precompute.

### Phase 3 Tasks

- Add a live pose dispatcher that consumes the shared streaming payload.
- Reuse existing PosePipeline batching/cache-writing machinery where it helps.
- Bypass generic frame replay/crop extraction when the live payload already has crops.
- Preserve pose cache writes and detected-properties integration needed by tracking.
- Keep replay fallback available for cache-only rebuilds.

### Phase 3 Files Expected

- `src/hydra_suite/core/tracking/pose_pipeline.py`
- `src/hydra_suite/core/tracking/worker.py`
- `src/hydra_suite/core/identity/pose/api.py`

### Phase 3 Exit Criteria

- Pose runs in the forward loop on filtered detections.
- Live pose outputs feed the same downstream orientation/visibility consumers.
- No separate pose precompute video pass is required for the normal forward workflow.

## Phase 4: Streaming CNN Integration On The Filtered Path

### Phase 4 Goal

Run CNN immediately alongside pose after head-tail reorientation using the same payload and cache semantics.

### Phase 4 Tasks

- Add a live CNN dispatcher parallel to the pose dispatcher.
- Replace `CNNPrecomputePhase` as the primary path for forward runs.
- Preserve or adapt `LiveCNNIdentityStore` and cache writer semantics.
- Add the emission point for a future `IdentityEvidenceCache` sidecar while keeping current top-1 caches intact.
- Ensure track-level history logic still reads the same label/confidence outputs.
- Support multiple configured CNN classifiers in one frame.

### Phase 4 Files Expected

- `src/hydra_suite/core/tracking/worker.py`
- `src/hydra_suite/core/tracking/live_features.py`
- `src/hydra_suite/core/tracking/precompute.py`
- `src/hydra_suite/core/identity/classification/cnn.py`

### Phase 4 Exit Criteria

- CNN no longer requires unified precompute for the default forward path.
- Multiple CNN models still work in one run.
- Track history/scoring remains behaviorally equivalent.

## Phase 5: Make Streaming The Default And Demote UnifiedPrecompute

### Phase 5 Goal

Change the forward YOLO OBB workflow so streaming pose/CNN analysis is the default behavior.

### Phase 5 Tasks

- Stop forcing batched unified precompute for pose/CNN in normal forward runs.
- Keep replay path behind an explicit fallback/compatibility decision.
- Ensure detection cache creation still happens where required for backward tracking and offline rebuilds.
- Ensure the identity-evidence sidecar is emitted equivalently for both live streaming and replay fallback.
- Keep AprilTag orchestration independent.
- Audit UI/workflow flags so the selected path is predictable and observable.

### Phase 5 Files Expected

- `src/hydra_suite/core/tracking/worker.py`
- `src/hydra_suite/trackerkit/gui/orchestrators/config.py`
- `src/hydra_suite/trackerkit/gui/orchestrators/session.py`
- `src/hydra_suite/trackerkit/obb_runtime_probe.py`

### Phase 5 Exit Criteria

- The default forward path no longer pays the extra precompute video pass for pose/CNN.
- Realtime and non-realtime forward workflows use the same primary analysis architecture.
- Replay remains available but no longer drives normal execution.

## Phase 6: Reduce UnifiedPrecompute To Replay/Recovery Mode

### Phase 6 Goal

Keep only the subset of unified precompute that remains operationally useful.

### Phase 6 Tasks

- Remove pose/CNN assumptions from the “normal path” code.
- Retain replay functionality for:
  - explicit cache regeneration
  - debug reproduction
  - post-hoc rebuilds from detection cache
- Keep AprilTag precompute isolated if still useful.
- Simplify or delete generic abstractions that no longer serve active paths.

### Phase 6 Files Expected

- `src/hydra_suite/core/tracking/precompute.py`
- `src/hydra_suite/core/tracking/worker.py`
- `src/hydra_suite/trackerkit/gui/dialogs/benchmark_dialog.py`
- any replay/debug entry points that call unified precompute explicitly

### Phase 6 Exit Criteria

- UnifiedPrecompute is no longer on the hot path for pose/CNN forward runs.
- Remaining code has a clear reason to exist.

## Phase 7: Revisit Interpolation And Offline Gaps

### Phase 7 Goal

Decide whether the occlusion interpolation worker should remain separate or adopt the same streaming/runtime abstractions later.

### Phase 7 Tasks

- Keep interpolation worker behavior unchanged during earlier phases.
- Reevaluate whether CNN/pose backend init and crop transport in the interpolation post-pass can share the new runtime abstractions.
- Do not merge this work into earlier phases unless needed for correctness.

### Phase 7 Files Expected

- `src/hydra_suite/trackerkit/gui/workers/crops_worker.py`

### Phase 7 Exit Criteria

- Interpolation is consciously either left separate or scheduled as a follow-on refactor.

## Risk Register

### Risk 1: Behavioral Drift Between CPU And GPU CNN Paths

Mitigation:

- add parity tests across CPU and CUDA entry points
- compare class names and confidences within expected tolerance
- test multi-head and single-head models

### Risk 2: Runtime Feature Fragmentation

Mitigation:

- define one routing table for pose/CNN runtime capability
- do not let worker logic guess backend support independently

### Risk 3: Cache Contract Breakage

Mitigation:

- preserve detection cache field ordering and semantics
- keep pose/CNN cache schemas stable during migration
- add compatibility probes before changing writers

### Risk 4: Realtime Throughput Regression From Over-Synchronization

Mitigation:

- keep async writers and bounded batching
- avoid per-frame blocking unless required by a specific backend
- measure queueing separately from model time

### Risk 5: MPS Gains Are Smaller Than CUDA Gains

Mitigation:

- treat “remove replay pass” as the first MPS win
- do not promise fully GPU-native crop flow on MPS in early phases

## Testing Strategy

### Unit Tests

- CNN CPU vs CUDA output parity
- pose CPU vs CUDA path parity where supported
- runtime selection tests for CPU/MPS/CUDA/ONNX/TensorRT
- shared payload shape and invariants
- cache writer schema compatibility tests

### Integration Tests

- forward YOLO OBB run with head-tail + pose enabled
- forward YOLO OBB run with head-tail + one CNN enabled
- forward YOLO OBB run with head-tail + pose + multiple CNNs enabled
- non-realtime forward run should not trigger precompute replay for pose/CNN default path
- replay fallback should still rebuild artifacts from detection cache

### Performance Validation

- compare before/after on real videos using `obb_runtime_probe.py` or `benchmark_trackerkit_obb.py`
- report:
  - end-to-end forward runtime
  - frame read time
  - crop extraction time
  - pose latency
  - CNN latency
  - cache write latency
  - GPU memory high-water mark where available

## Recommended Delivery Order

If both this plan and the identity overhaul are being executed together, the best combined order is:

1. Streaming Phase 0: instrumentation, guardrails, and artifact-contract audit.
2. Streaming Phase 1: shared oriented-analysis payload with stable detection-slot indexing.
3. Streaming Phase 2: GPU-native CNN runtime path plus posterior-output hook.
4. Identity Overhaul Phase 0: posterior-preserving evidence contracts and sidecar cache, implemented on top of the shared streaming payload.
5. Streaming Phase 3 and Phase 4: live pose and CNN dispatch on the filtered path, including evidence-sidecar emission in both live and replay modes.
6. Identity Overhaul Phase 1 and Phase 2: online decoder, commitment, and slot reservation using the new live evidence stream.
7. Streaming Phase 5: make the streaming path the default forward path.
8. Identity Overhaul Phase 3 and Phase 4: offline smoothing and global fragment assignment using the preserved evidence sidecar.
9. Streaming Phase 6 and Identity Overhaul Phase 5: retire legacy replay-first and hard-label identity heuristics.
10. Streaming Phase 7: revisit interpolation and any remaining replay-only gaps.

This order is preferred because it prevents the identity overhaul from being built on top of a transport path that is about to be replaced. The streaming plan establishes the runtime and cache substrate first; the identity overhaul then uses that substrate for online and offline decoding.

## Final Integrated Product

When both specs are complete, the final product should behave as follows:

- forward YOLO OBB runs dispatch pose and CNN analysis directly from the filtered, canonicalized detection stream
- the same live and replay code paths emit a stable identity evidence sidecar alongside existing compatibility caches
- the live tracker uses posterior-aware identity decoding with uniqueness-aware visible assignment and slot reservation
- the offline post-processing path consumes preserved evidence rather than flattened top-1 labels
- replay remains as a recovery and regeneration mechanism, not the primary path for normal CNN and pose analysis

## Immediate Next Steps

1. Add the shared streaming analysis payload type and wire it at the filtered-detection boundary.
2. Add `CNNIdentityBackend.predict_batch_cuda()` plus a posterior-producing hook and parity tests.
3. Define the stable detection-slot indexing and cache-emission contract that the identity sidecar will use.
4. Refactor pose and CNN live dispatch to consume the same payload directly.
5. Only after those pieces exist, start the identity evidence cache and online decoder work from `trackerkit-identity-overhaul-spec.md`.
