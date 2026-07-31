# Inference Pipeline Redesign

**Date:** 2026-04-26
**Status:** Approved for implementation (revised 2026-05-04 with downstream-consumer audit fixes)
**Replaces:** `detection_phase.py`, `precompute.py`, `pose_pipeline.py`, scattered identity precompute code

---

## Audit Revision Notes (2026-05-04)

A downstream-consumer audit found that the original spec, executed verbatim, would break the build at module load time and lose data the existing pipeline relies on. The following revisions are now incorporated throughout this document. Each is referenced by the section that resolves it.

| # | Issue | Resolution |
|---|---|---|
| 1 | `detection_ids` (legacy `frame_idx * 10000 + slot`) was not in any result type but is used as a primary key by CSV writer, identity evidence, pose keypoint maps, and AprilTag association | Added to `OBBResult` (see Result Types) |
| 2 | `OnlineIdentityDecoder` needs full calibrated probability vectors per detection, not just top-1 confidence | Clarified that consumers access `cnn_factors[i].calibrated_probabilities` directly; `resolved_label`/`resolved_confidence` are CSV/visualization-only convenience fields (see Identity Evidence Layer) |
| 3 | `core/identity/pose/features.py` (kept) reads from deleted `IndividualPropertiesCache` | Added to "Affected kept files" with a rewire to read from new `PoseCache` |
| 4 | `core/identity/properties/export.py` (kept) imports `CNNIdentityCache` and `DetectedPropertiesCache` | Added to "Affected kept files" with rewires to new caches |
| 5 | `core/identity/dataset/oriented_video.py` (kept) imports deleted `DetectionCache` | Added to "Affected kept files" |
| 6 | `core/tracking/optimizer.py` and `optimizer_workers.py` (kept) import `DetectionFilter` and `DetectionCache` from deleted modules | Added to "Affected kept files"; filter logic exposed as `apply_detection_filter` shim |
| 7 | `posekit/gui/workers.py` (kept) lazy-imports from deleted `core/identity/pose/api.py` | Added to "Affected kept files" with a small public helper in `core/inference/api.py` |
| 8 | `core/detectors/__init__.py`, `core/identity/classification/__init__.py`, `core/identity/pose/__init__.py`, `core/identity/properties/__init__.py`, `data/__init__.py` re-export deleted symbols and crash on import the moment deletion happens | Added to "Affected kept files"; `__init__.py` updates moved to a dedicated migration step *before* deletion |
| 9 | Backward pass pseudocode invented `_run_backward_pass()` / `_run_consensus_resolution()` methods that don't exist; backward pass is actually a separate `TrackingWorker(backward_mode=True)` launched by the GUI | Rewrote "Non-RT flow" with the actual GUI-orchestrated architecture |
| 10 | Old CNN cache stored *post-calibration* probabilities; new CNN cache stores *raw*. Without invalidation, the feature-flag coexistence window silently corrupts identity decoding | Added `CACHE_SCHEMA_VERSION` field to `CacheKey`; mismatch invalidates all legacy caches |
| 11 | `HeadTailResult.canonical_affines` cannot be reconstructed from a cache load | Made `canonical_affines: np.ndarray \| None`; cache path returns `None` |
| 12 | `StreamingAnalysisPayload` (kept as-is) had no constructor in the new pipeline | Moved to "Affected kept files" with construction site in `runner.run_realtime()` |
| 13 | Tests importing deleted modules are unaccounted for | Migration Strategy now includes a dedicated test-migration step listing the affected test files |
| 14 | `canvas_dims` and `M_inverse` fields were silently dropped from `DetectionCache` | Documented as deprecated in the migration notes — no current consumer reads them, but the deprecation is now explicit |

---

## Motivation

The current inference pipeline is fragmented across a dozen files with no clean boundary between inference and tracking. Config is an untyped flat dict. The same model-loading and inference logic is duplicated across streaming and batched paths. Mode branching (`if realtime`, `if backward`, `if preview`) is scattered throughout a 4,252-line `worker.py`. There is no single place where a developer can trace what happens to a frame from read to cached result.

This redesign introduces `core/inference/` as a self-contained module with two modes, two runtime paths, a typed config schema, a producer-consumer pipeline, and per-type caches with automatic invalidation.

---

## Design Goals

1. One clear code path per mode (RT vs non-RT) readable top to bottom
2. Two runtime paths (CUDA GPU-first, CPU/MPS CPU-first) with no scattered `if cuda:` branches
3. Typed config schema — JSON on disk, dataclasses in memory, validated at load time
4. Per-type sidecar caches with automatic invalidation (path + mtime); partial reruns when only one model changes
5. `worker.py` delegates all inference to `InferenceRunner`; tracking logic is untouched

---

## Two Modes

### Realtime (RT) mode — `config.realtime = True`

Frame-by-frame. No caches read or written. All individual analysis (HeadTail, CNN, Pose) is limited to crops from the current frame, batched within the frame via `ThreadPoolExecutor`. The tracking loop in `worker.py` calls `runner.run_realtime(frame)` per frame and immediately runs Kalman + assignment on the returned `FrameResult`.

### Non-RT mode — `config.realtime = False`

Two chained stages, automatic:

1. **Inference pass** — `runner.run_batch_pass(video, cache_dir)`: pure inference, no tracking. Runs the full producer-consumer pipeline, writes per-type sidecar caches. Skipped entirely if all caches are valid and `config.use_cache = True`.
2. **Tracking pass** — `worker.py` reads from caches via `runner.load_frame(frame_idx)`, runs Kalman + assignment, then backward pass and consensus resolution as today.

Batch size is defined by `InferenceConfig.detection_batch_size`. If set to 4, OBB detection runs on 4 frames at once, then all detections from those 4 frames are batched together for HeadTail, CNN, and Pose cross-frame batching.

---

## Two Runtime Paths

Runtime is defined per inference type via `ComputeRuntime`. Mixing CUDA-group and CPU-group runtimes in the same config is a validation error.

```
CUDA group:  cuda | onnx_cuda | tensorrt
CPU group:   cpu  | mps       | onnx_cpu | onnx_coreml
```

### CUDA path (`cuda_mode = True`)

```
NVDEC → CUDA tensor frame
  → OBB (CUDA / ONNX_CUDA / TensorRT)
  → Filtering
  → Crop extraction on GPU (affine, native resolution)
  ├── HeadTail inference (resize → CUDA/ONNX_CUDA/TensorRT)
  ├── CNN inference     (resize → CUDA/ONNX_CUDA/TensorRT)   [concurrent]
  ├── Pose inference    (resize → CUDA/ONNX_CUDA/TensorRT)
  └── AABB CPU crops (single .cpu().numpy() pull per batch)
        → AprilTag inference (always CPU)
```

### CPU/MPS path (`cuda_mode = False`)

```
CPU video read → numpy frame
  → OBB (CPU / MPS / ONNX_CPU / ONNX_CoreML)
  → Filtering
  → Canonical affine crops via cv2.warpAffine → CPU tensor
  ├── HeadTail inference (CPU/MPS)
  ├── CNN inference      (CPU/MPS)                            [concurrent]
  ├── Pose inference     (CPU/MPS)
  └── AABB CPU crops → AprilTag inference (CPU)
```

AprilTag always runs on CPU regardless of runtime path.

---

## Module Structure

```
src/hydra_suite/core/inference/
    __init__.py              # public API: InferenceRunner, InferenceConfig, InferenceResult
    config.py                # all typed dataclasses (see Config Schema section)
    runtime.py               # RuntimeContext, CUDA_RUNTIMES, CPU_RUNTIMES
    runner.py                # InferenceRunner: model lifecycle, run_realtime, run_batch_pass
    result.py                # FrameResult, OBBResult, HeadTailResult, CNNResult,
                             # PoseResult, AprilTagResult, FrameIdentityEvidence
    stages/
        __init__.py
        obb.py               # run_obb(frames, models, config, runtime) -> list[OBBResult]
        filtering.py         # filter_detections(raw, config, roi_mask) -> OBBResult
        crops.py             # extract_canonical_crops(), extract_aabb_crops()
        headtail.py          # run_headtail(crops, obb, model, config, runtime) -> HeadTailResult
        cnn.py               # run_cnn(crops, obb, model, config, runtime) -> CNNResult
        pose.py              # run_pose(crops, obb, model, config, runtime) -> PoseResult
        apriltag.py          # run_apriltag(cpu_crops, obb, model, config) -> AprilTagResult
    cache/
        __init__.py
        base.py              # CacheHandle ABC, CacheKey, mtime-based invalidation
        detection.py         # DetectionCache: OBBResult + HeadTailResult per frame
        cnn.py               # CNNCache: CNNResult per frame, one file per phase label
        pose.py              # PoseCache: PoseResult per frame
        apriltag.py          # AprilTagCache: AprilTagResult per frame
```

Everything outside `core/inference/` that the old pipeline owned is deleted after this is verified working (see Migration section).

---

## Config Schema

JSON on disk. Loaded into typed dataclasses by `InferenceConfig.from_json()`. Validated at load time — runtime consistency check fires before any model is loaded.

```python
ComputeRuntime = Literal["cpu", "mps", "cuda", "onnx_cpu", "onnx_cuda", "onnx_coreml", "tensorrt"]

CUDA_RUNTIMES = frozenset({"cuda", "onnx_cuda", "tensorrt"})
CPU_RUNTIMES  = frozenset({"cpu", "mps", "onnx_cpu", "onnx_coreml"})


@dataclass
class OBBDirectConfig:
    model_path: str
    compute_runtime: ComputeRuntime = "cpu"
    confidence_floor: float = 1e-3        # raw inference floor before filtering
    confidence_threshold: float = 0.25   # post-inference filter gate


@dataclass
class OBBSequentialConfig:
    detect_model_path: str
    obb_model_path: str
    detect_compute_runtime: ComputeRuntime = "cpu"   # stage-1 separate runtime
    obb_compute_runtime: ComputeRuntime = "cpu"      # stage-2 separate runtime
    detect_confidence_threshold: float = 1e-3
    obb_confidence_threshold: float = 1e-3           # IOU always 1.0 in sequential mode
    detect_image_size: int = 0                       # 0 = auto
    crop_pad_ratio: float = 0.15
    min_crop_size_px: float = 64.0
    enforce_square_crop: bool = True
    stage2_image_size: int = 160
    stage2_batch_size: int | None = None


@dataclass
class OBBConfig:
    mode: Literal["direct", "sequential"] = "direct"
    direct: OBBDirectConfig | None = None
    sequential: OBBSequentialConfig | None = None
    target_classes: list[int] = field(default_factory=list)
    max_detections: int = 20
    min_object_size: float = 0.0
    max_object_size: float = float("inf")
    confidence_threshold: float = 0.25    # post-inference filter applied after raw detection
    iou_threshold: float = 0.45           # NMS threshold; direct mode only


@dataclass
class HeadTailConfig:
    model_path: str
    compute_runtime: ComputeRuntime = "cpu"
    confidence_threshold: float = 0.5
    candidate_confidence_threshold: float | None = None  # pre-filter before crop extraction;
                                                          # defaults to OBBConfig.confidence_threshold
    batch_size: int = 64
    canonical_aspect_ratio: float = 2.0
    canonical_margin: float = 1.3


@dataclass
class CNNConfig:
    label: str                             # phase name, e.g. "identity"
    model_path: str
    compute_runtime: ComputeRuntime = "cpu"
    confidence_threshold: float = 0.5     # applied at tracking time, not cached
    batch_size: int = 64
    scoring_mode: Literal["atomic", "per_head_average"] = "atomic"
    match_bonus: float = 0.1
    mismatch_penalty: float = 0.3
    calibration_temperature: float = 1.0  # applied at tracking time, not cached


@dataclass
class PoseYOLOConfig:
    model_path: str
    compute_runtime: ComputeRuntime = "cpu"
    confidence_threshold: float = 1e-4
    iou_threshold: float = 0.7
    max_detections_per_crop: int = 1
    batch_size: int = 64


@dataclass
class PoseSLEAPConfig:
    model_path: str
    compute_runtime: ComputeRuntime = "cpu"
    conda_env: str = "sleap"
    batch_size: int = 4
    max_instances: int = 1


@dataclass
class PoseConfig:
    backend: Literal["yolo", "sleap"] = "yolo"
    skeleton_file: str = ""
    yolo: PoseYOLOConfig | None = None
    sleap: PoseSLEAPConfig | None = None
    crop_padding: float = 0.1
    suppress_foreign_regions: bool = True
    background_color: tuple[int, int, int] = (0, 0, 0)
    anterior_keypoints: list[str] = field(default_factory=list)
    posterior_keypoints: list[str] = field(default_factory=list)
    ignore_keypoints: list[str] = field(default_factory=list)
    min_keypoint_confidence: float = 0.2
    min_valid_keypoints: int = 1
    overrides_headtail: bool = True


@dataclass
class AprilTagConfig:
    enabled: bool = False
    tag_family: str = "tag36h11"
    threads: int = 4
    max_hamming: int = 1
    decimate: float = 1.0
    blur: float = 0.8
    refine_edges: bool = True
    decode_sharpening: float = 0.25
    unsharp_kernel: tuple[int, int] = (5, 5)
    unsharp_sigma: float = 1.0
    unsharp_amount: float = 1.5
    contrast_factor: float = 1.5
    max_tag_id: int | None = None
    crop_padding: float = 0.1


@dataclass
class InferenceConfig:
    obb: OBBConfig
    headtail: HeadTailConfig | None = None
    cnn_phases: list[CNNConfig] = field(default_factory=list)
    pose: PoseConfig | None = None
    apriltag: AprilTagConfig = field(default_factory=AprilTagConfig)
    detection_batch_size: int = 1          # frames per OBB batch; drives cross-frame batching
                                           # for all individual analysis stages
    realtime: bool = False
    use_cache: bool = True
    cache_dir: str | None = None           # defaults to video directory if None

    @staticmethod
    def from_json(path: str) -> "InferenceConfig": ...
    def to_json(self, path: str) -> None: ...
    def _validate_runtime_consistency(self) -> None:
        # Collect all ComputeRuntime values from all sub-configs.
        # Raises InferenceConfigError if any runtime spans both groups.
        ...
    def _collect_all_runtimes(self) -> set[ComputeRuntime]: ...
```

### Runtime validation rule

At `from_json()` load time:

```python
uses_cuda = any(r in CUDA_RUNTIMES for r in runtimes)
uses_cpu  = any(r in CPU_RUNTIMES  for r in runtimes)
if uses_cuda and uses_cpu:
    raise InferenceConfigError(...)
```

Frame reading (NVDEC vs direct) is derived automatically from whether any runtime is in `CUDA_RUNTIMES`. No separate flag needed.

---

## RuntimeContext

Built once from `InferenceConfig`. Passed into every stage function. Stage functions never probe hardware directly.

```python
@dataclass(frozen=True)
class RuntimeContext:
    cuda_mode: bool
    device: str              # "cuda:0", "mps", or "cpu"
    use_nvdec: bool          # cuda_mode AND nvdec available
    default_runtime: ComputeRuntime

    @staticmethod
    def from_config(config: InferenceConfig) -> "RuntimeContext":
        runtimes = config._collect_all_runtimes()
        cuda_mode = any(r in CUDA_RUNTIMES for r in runtimes)
        device = _resolve_device(cuda_mode)
        nvdec = cuda_mode and _nvdec_available()
        default = "cuda" if cuda_mode else "cpu"
        return RuntimeContext(cuda_mode, device, nvdec, default)
```

---

## InferenceRunner

Owns model lifecycle. Exposes two methods to `worker.py`. All pipeline complexity (queues, workers, buffers) is hidden inside `runner.py`.

```python
class InferenceRunner:
    def __init__(self, config: InferenceConfig, cache_dir: Path):
        self.config = config
        self.runtime = RuntimeContext.from_config(config)
        self.cache_dir = cache_dir
        self._models = _load_all_models(config, self.runtime)

    def caches_all_valid(self) -> bool:
        caches = _open_caches(self.config, self.cache_dir, self._models)
        return self.config.use_cache and all(c.is_valid() for c in caches.all())

    def run_realtime(self, frame: np.ndarray | torch.Tensor) -> FrameResult:
        """Single frame → typed result. No I/O. Caller drives loop."""
        obb = run_obb([frame], self._models.obb, self.config.obb, self.runtime)[0]
        obb = filter_detections(obb, self.config.obb, roi_mask=None)
        gpu_crops, cpu_crops = _extract_crops(frame, obb, self.config, self.runtime)
        with ThreadPoolExecutor() as ex:
            f_ht   = ex.submit(run_headtail, gpu_crops, obb, self._models.headtail,
                               self.config.headtail, self.runtime)
            f_cnns = [ex.submit(run_cnn, gpu_crops, obb, self._models.cnn[i],
                                phase, self.runtime)
                      for i, phase in enumerate(self.config.cnn_phases)]
            f_pose = ex.submit(run_pose, gpu_crops, obb, self._models.pose,
                               self.config.pose, self.runtime)
            f_tag  = ex.submit(run_apriltag, cpu_crops, obb, self._models.apriltag,
                               self.config.apriltag)
        return _assemble_frame_result(
            obb, f_ht.result(), [f.result() for f in f_cnns],
            f_pose.result(), f_tag.result(), self.config
        )

    def run_batch_pass(self, video_path: Path,
                       progress_cb: Callable[[int], None] | None = None
                       ) -> InferencePassResult:
        """Full forward inference pass. Writes per-type caches. Chains automatically
        before the tracking pass in non-RT mode."""
        ...

    def load_frame(self, frame_idx: int) -> FrameResult:
        """Load a cached FrameResult for the tracking pass. No inference."""
        ...

    def close(self) -> None: ...
```

### Non-RT producer-consumer pipeline

`run_batch_pass` builds and starts these workers:

```
FrameReaderWorker  → frame_q
OBBBatchWorker     → obb_q        accumulates OBB batch size; runs OBB + filtering
CropWorker         → crop_q       extracts canonical GPU crops + AABB CPU crops from batch
                   → cpu_crop_q   AABB crops fed to AprilTag in parallel
IndividualWorker   → results_q    HeadTail + CNN phases + Pose concurrently
                                  via ThreadPoolExecutor; shares crop tensors read-only
AprilTagWorker     → results_q    CPU; concurrent with IndividualWorker
ResultMergeWorker  → cache_q      assembles per-frame FrameResult
CacheWriterWorker               async disk writes; never stalls inference
```

At any point: `OBBBatchWorker` processes batch N+1 while `IndividualWorker` processes crops from batch N while `CacheWriterWorker` writes batch N-1. GPU stays saturated.

---

## Stage Functions

Plain functions in `stages/`. No classes, no hidden state. Models and config are arguments; typed results are return values. No I/O, no mode branching, no device detection.

```python
# stages/obb.py
def run_obb(frames, models, config, runtime) -> list[OBBResult]:
    # Direct: model.predict(frames) → extract corners, centroids, angles, confidences
    # Sequential: stage1 → crop → stage2

# stages/filtering.py
def filter_detections(raw, config, roi_mask) -> OBBResult:
    # confidence gate → size gate → aspect ratio gate → ROI mask → NMS

# stages/crops.py
def extract_canonical_crops(frame, obb_result, config, runtime) -> torch.Tensor:
    # GPU path: kornia/grid_sample affine on CUDA tensor
    # CPU path: cv2.warpAffine per crop → torch.from_numpy (CPU tensor)
    # Always returns a tensor; stage functions are device-agnostic

def extract_aabb_crops(frame, obb_result, padding) -> list[np.ndarray]:
    # Always CPU numpy; for AprilTag only

# stages/headtail.py
def run_headtail(crops, obb_result, model, config, runtime) -> HeadTailResult:
    # Resize crops to model input size → ClassifierBackend.predict_batch()
    # Map label (up/down/left/right) → heading angle relative to OBB axis

# stages/cnn.py
def run_cnn(crops, obb_result, model, config, runtime) -> CNNResult:
    # Resize crops → predict_batch() → raw per-factor probability vectors
    # Temperature calibration and scoring_mode applied at tracking time

# stages/pose.py
def run_pose(crops, obb_result, model, config, runtime) -> PoseResult:
    # Resize crops → model.predict() → extract keypoints (K, 3) per detection

# stages/apriltag.py
def run_apriltag(cpu_crops, obb_result, model, config) -> AprilTagResult:
    # Preprocess (unsharp, contrast) → detect tags → filter by max_tag_id
```

Three rules enforced for all stage functions:
1. **No I/O** — stage functions never read or write disk
2. **No mode branching** — no knowledge of RT vs non-RT
3. **No device detection** — `RuntimeContext` tells them where tensors live

Heading resolution (pose overrides headtail priority) happens in `result.py` when assembling `FrameResult`, not inside any stage function.

---

## Result Types

```python
@dataclass
class OBBResult:
    frame_idx: int
    centroids: np.ndarray        # (D, 2) cx, cy
    angles: np.ndarray           # (D,) radians
    sizes: np.ndarray            # (D,) area px²
    shapes: np.ndarray           # (D, 2) ellipse_area, aspect_ratio
    confidences: np.ndarray      # (D,) raw detection confidence
    corners: np.ndarray          # (D, 4, 2) OBB corners
    detection_ids: np.ndarray    # (D,) int64; primary key used downstream
                                 # (CSV rows, identity evidence, tag association,
                                 #  pose keypoint maps). Generated as
                                 #  frame_idx * DETECTION_ID_STRIDE + slot.


@dataclass
class HeadTailResult:
    heading_hints: np.ndarray              # (D,) radians; nan = no confident direction
    heading_confidences: np.ndarray        # (D,)
    directed_mask: np.ndarray              # (D,) uint8; 1 = heading trusted
    canonical_affines: np.ndarray | None   # (D, 2, 3) affine matrices, or None
                                           #  when loaded from cache (cache stores
                                           #  outputs only; affines are recomputable
                                           #  from OBBResult + canonical_aspect_ratio
                                           #  + canonical_margin if needed).


@dataclass
class CNNFactorPrediction:
    factor_name: str
    class_names: list[str]
    raw_probabilities: np.ndarray   # (num_classes,) pre-calibration


@dataclass
class CNNDetectionPrediction:
    det_index: int
    factors: list[CNNFactorPrediction]  # len=1 flat model; len=K multi-head


@dataclass
class CNNResult:
    label: str                                  # from CNNConfig.label
    predictions: list[CNNDetectionPrediction]   # one per detection


@dataclass
class PoseResult:
    keypoints: np.ndarray    # (D, K, 3): [x, y, confidence] per keypoint per detection
    valid_mask: np.ndarray   # (D,) bool: meets min_keypoint_confidence + min_valid_keypoints


@dataclass
class AprilTagResult:
    tag_ids: list[int]
    det_indices: list[int]   # which OBB detection each tag is associated with
    centers: np.ndarray      # (T, 2)
    corners: np.ndarray      # (T, 4, 2)


@dataclass
class FrameResult:
    frame_idx: int
    obb: OBBResult
    filtered_indices: list[int]       # detections that survived filtering
    headtail: HeadTailResult | None
    cnn: list[CNNResult]              # one per CNN phase
    pose: PoseResult | None
    apriltag: AprilTagResult | None
    resolved_headings: np.ndarray     # (D,) final merged heading per detection
                                      # pose → headtail → OBB axis (priority in result.py)
```

---

## Cache System

Separate sidecar `.npz` file per inference type. Invalidated by `CacheKey` mismatch.

### Invalidation principle

Cache raw model outputs (pre-threshold). Filtering, calibration, scoring mode, and heading resolution are all re-applied at tracking time. This means threshold changes never cause a cache miss — only model or crop extraction parameter changes do.

```python
# Bumped any time the on-disk schema changes. Caches written with a different
# schema_version are invalidated automatically. Increment whenever a field is
# added/removed/renamed in any cached result type.
CACHE_SCHEMA_VERSION = 2  # v1 = legacy pre-redesign caches; v2 = new pipeline


@dataclass
class CacheKey:
    schema_version: int     # CACHE_SCHEMA_VERSION at write time
    model_path: str
    model_mtime: float      # os.path.getmtime at runner startup
    config_hash: str        # hash of fields that change what the model sees (not thresholds)
```

`schema_version` is checked first; a mismatch invalidates the cache regardless of model identity. This guarantees that legacy caches written by the pre-redesign pipeline (which stored already-calibrated CNN probabilities, baked thresholds into the detection cache, etc.) are never re-used incorrectly during the feature-flag coexistence window.

### Per-type cache keys

| Cache | Key fields |
|---|---|
| `DetectionCache` | schema_version + OBB model path + mtime (thresholds excluded — re-applied at tracking time) |
| `HeadTailCache` | schema_version + HT model path + mtime + `hash(canonical_aspect_ratio, canonical_margin)` |
| `CNNCache` (per phase) | schema_version + CNN model path + mtime (calibration_temperature, scoring_mode excluded — applied at tracking time inside `IdentityEvidenceBuilder`) |
| `PoseCache` | schema_version + Pose model path + mtime + `hash(crop_padding, suppress_foreign_regions, background_color)` |
| `AprilTagCache` | schema_version + `hash(tag_family, decimate, blur, refine_edges, unsharp_kernel, unsharp_sigma, unsharp_amount, contrast_factor)` |
| `IdentityEvidenceCache` | schema_version + CNN model path + mtime + AprilTag config hash + `IdentityCatalog` version |

**Calibration semantics changed:** the legacy `CNNIdentityCache` stored *post-calibration* probabilities (calibration was baked into the cache hash). The new `CNNCache` stores *raw* pre-calibration probabilities, and `IdentityEvidenceBuilder` applies temperature at tracking time. Without the schema bump, switching pipelines would either double-apply calibration (legacy cache → new builder applies again) or skip it (new cache → old consumer expects calibrated values). The version bump invalidates legacy caches on first run of the new pipeline.

### Partial reruns

Invalid caches are overwritten; valid caches are untouched. Changing the HeadTail model only re-runs HeadTail. Changing a CNN phase model only re-runs that phase. `CacheWriterWorker` skips writing to a cache that was already valid at startup.

### Cache handle contract

```python
class CacheHandle(ABC):
    def is_valid(self) -> bool: ...           # key match check
    def write_frame(self, frame_idx, result): ...
    def read_frame(self, frame_idx): ...
    def close(self): ...
```

---

## worker.py Interface

`worker.py` imports only from `core/inference/`. It never touches queues, stage functions, or cache handles directly.

### Non-RT flow

```python
def run(self):
    config = InferenceConfig.from_json(self.params["config_path"])
    runner = InferenceRunner(config, cache_dir=self._resolve_cache_dir())

    if not runner.caches_all_valid():
        runner.run_batch_pass(self.video_path, progress_cb=self._emit_progress)

    reader = FrameReader.open(self.video_path)
    for frame_idx, frame in enumerate(reader):
        frame_result = runner.load_frame(frame_idx)
        evidence = self._identity_builder.build(frame_result)
        tracks = self.tracker.update(frame_result, evidence)
        self._emit_frame(frame_idx, frame, tracks)

    runner.close()
```

**Backward pass and consensus resolution are NOT methods on this worker.** They are orchestrated by the GUI layer (`trackerkit/gui/main_window.py`), which:

1. Launches the forward `TrackingWorker` (the one shown above).
2. On completion, launches a second `TrackingWorker(backward_mode=True)` instance. The backward worker reuses the forward worker's `cache_dir`; it never re-runs inference. Its `run()` method follows the exact pseudocode above, except `runner.caches_all_valid()` is asserted (cache MUST exist — backward mode refuses to run if any required cache is missing) and frame iteration is reversed.
3. Runs the consensus resolution step on the merged forward+backward CSV, in-process inside the GUI (this lives in `core/post/identity_postprocess.py` and operates on DataFrames, not on inference state).

The `InferenceRunner` is therefore stateless across forward/backward: both workers construct their own `InferenceRunner` from the same config + cache_dir; the second one finds all caches valid and skips straight to `load_frame()`. No special "backward mode" wiring is needed in `core/inference/`.

`load_frame(frame_idx)` MUST return identical `FrameResult` objects for forward and backward callers — this is the single contract that backward-pass correctness depends on. It is enforced by reading from caches only (no live inference inside `load_frame`).

### RT flow

```python
def run(self):
    config = InferenceConfig.from_json(self.params["config_path"])
    runner = InferenceRunner(config, cache_dir=None)

    reader = FrameReader.open(self.video_path)
    for frame_idx, frame in enumerate(reader):
        frame_result = runner.run_realtime(frame)
        evidence = self._identity_builder.build(frame_result)
        tracks = self.tracker.update(frame_result, evidence)
        self._emit_frame(frame_idx, frame, tracks)

    runner.close()
```

### FrameResult → tracking

`worker.py` reads from `FrameResult`:
- `frame_result.filtered_indices` + `frame_result.obb` → Kalman measurements
- `frame_result.obb.detection_ids` → primary key for CSV rows, identity evidence,
  pose keypoint maps, AprilTag association (replaces the legacy
  `frame_idx * 10000 + det_slot` integer used throughout the old worker loop)
- `frame_result.resolved_headings` → measurement theta
- `frame_result.cnn` → fed to `IdentityEvidenceBuilder`
- `frame_result.apriltag` → fed to `IdentityEvidenceBuilder`
- `frame_result.pose` → keypoints written to CSV

No dict lookups, no magic string keys anywhere in tracking code.

---

## Identity Evidence Layer

Sits between `FrameResult` (raw inference) and the four downstream identity consumers. Built in `worker.py` before assignment each frame. Lives in `core/tracking/identity/evidence.py`.

```python
@dataclass
class CNNFactorEvidence:
    factor_name: str
    class_names: list[str]
    calibrated_probabilities: np.ndarray   # post-temperature softmax
    winning_class: str
    confidence: float


@dataclass
class DetectionIdentityEvidence:
    det_index: int
    cnn_factors: list[CNNFactorEvidence] | None   # None if no CNN phase for identity
    apriltag_label: str | None                    # mapped from tag_id via IdentityCatalog
    apriltag_tag_id: int | None
    resolved_label: str | None                    # AprilTag overrides CNN when both present
    resolved_confidence: float
    is_authoritative: bool                        # True = from AprilTag


@dataclass
class FrameIdentityEvidence:
    frame_idx: int
    phase_label: str
    detections: list[DetectionIdentityEvidence]


class IdentityEvidenceBuilder:
    """One instance per identity-labeled CNN phase. worker.py creates one builder
    per phase in config.cnn_phases and calls build() on each per frame."""
    def __init__(self, config: CNNConfig, catalog: IdentityCatalog): ...
    def build(self, frame_result: FrameResult) -> FrameIdentityEvidence:
        # 1. Apply calibration_temperature to raw_probabilities for this phase
        # 2. Apply scoring_mode aggregation across factors
        # 3. Map AprilTag tag_ids → identity labels via catalog
        # 4. Resolve: AprilTag overrides CNN when both present
        ...
```

### Downstream consumers of `FrameIdentityEvidence`

| Consumer | What it receives | Specific fields read | When |
|---|---|---|---|
| `TrackAssigner` | `FrameIdentityEvidence` | per detection: `cnn_factors[*].calibrated_probabilities` (full vector), `apriltag_label`, `is_authoritative` | Before assignment — adjusts cost matrix via `match_bonus` / `mismatch_penalty` |
| `OnlineIdentityDecoder` | `DetectionIdentityEvidence` keyed by `track_id` (post-assignment) | `cnn_factors[*].calibrated_probabilities` and `class_names` (the **full posterior** is required for catalog remapping in `_remap_source_log_probs_to_catalog`); AprilTag fields when authoritative | After assignment — Bayesian log-posterior update per track |
| `IdentityEvidenceCache` | `FrameIdentityEvidence` | All fields verbatim | Written per frame for offline decoder + post-processing |
| CSV writer | Committed label + confidence per track | `resolved_label`, `resolved_confidence` only | End of tracking loop |

**Important:** `DetectionIdentityEvidence.resolved_label` / `resolved_confidence` are convenience top-1 fields for the CSV writer and visualization. The `OnlineIdentityDecoder` does **not** use them — it consumes `cnn_factors[i].calibrated_probabilities` (the full distribution) per factor and remaps it into the catalog space. The Bayesian update path is therefore lossless; the top-1 fields are derived for downstream summarization only.

`IdentityEvidenceCache` key: CNN model path + mtime + AprilTag config hash + `IdentityCatalog` version + cache schema version. Changing the tag→identity mapping invalidates identity evidence but not raw CNN predictions.

`IdentityEvidenceBuilder` is the single place where temperature calibration, scoring mode aggregation, and AprilTag→CNN priority logic live. Nothing else performs these operations.

---

## Code Provenance

### Serves as direct inspiration (logic kept, structure rewritten)

| Existing file | New home | What is preserved |
|---|---|---|
| `detectors/yolo_detector.py` | `stages/obb.py` | OBB inference, sequential/direct mode logic |
| `detectors/_obb_geometry.py` | `stages/filtering.py` | NMS, IOU, corner geometry — reused nearly verbatim |
| `detectors/_direct_obb_runtime.py` | `stages/obb.py` + `runtime.py` | Direct CUDA/ONNX/TRT executor |
| `detectors/_runtime_artifacts.py` | Moved to `runtime/` | ONNX/TRT artifact export and validation |
| `tracking/detection_phase.py` | `runner.py` OBBBatchWorker | Batched detection pass concept |
| `tracking/pose_pipeline.py` | `runner.py` IndividualWorker + `stages/pose.py` | Double-buffered pipeline concept, ThreadPoolExecutor |
| `tracking/precompute.py` | `runner.py` pipeline workers | Phase orchestration concept |
| `identity/classification/headtail.py` | `stages/headtail.py` | Heading-from-direction, label normalization |
| `identity/classification/cnn.py` | `stages/cnn.py` + `cache/cnn.py` | Multi-head prediction logic |
| `identity/pose/backends/yolo.py` | `stages/pose.py` | YOLO pose inference |
| `identity/pose/backends/sleap.py` | `stages/pose.py` | SLEAP backend |
| `data/detection_cache.py` | `cache/detection.py` | NPZ format, per-frame read/write contract |
| `data/tag_observation_cache.py` | `cache/apriltag.py` | Tag cache format |
| `identity/properties/cache.py` + `detected_cache.py` | `cache/pose.py` | Pose properties cache format |
| `tracking/cnn_features.py` + `tag_features.py` + `evidence_emitter.py` | `tracking/identity/evidence.py` | Evidence building and AprilTag fusion logic |
| `tracking/live_features.py` | RT path in `runner.py` | In-memory store concept for RT mode |

### Kept entirely as-is (no changes)

- `core/filters/kalman.py`
- `core/assigners/hungarian.py`
- `core/post/processing.py` + `identity_postprocess.py`
- `identity/catalog.py`, `evidence.py`, `online.py`, `cache.py`
- `identity/geometry.py`, `fragment_solver.py`, `calibration.py`
- `identity/classification/backend.py` — `ClassifierBackend` shared kernel used by new stages
- `identity/classification/apriltag.py`, `errors.py`
- `identity/pose/quality.py`, `artifacts.py`, `types.py`
- `runtime/compute_runtime.py`
- `data/csv_writer.py`
- `tracking/visualization.py`, `profiler.py`, `orientation.py`
- `tracking/density.py`, `confidence_density.py`
- `detectors/bg_detector.py`, `bg_optimizer.py`

### Affected kept files (need targeted updates, not deletion)

These files survive the refactor but currently import from deleted modules. Each MUST be rewired before the deletion step or the import graph breaks at module load. The plan describes the rewire for each.

| File | Currently imports | Required change |
|---|---|---|
| `core/detectors/__init__.py` | re-exports `DetectionFilter`, `create_detector`, `YOLOOBBDetector` | Drop the three deleted re-exports; keep only `bg_detector`/`bg_optimizer` exports |
| `core/identity/classification/__init__.py` | re-exports `CNNIdentityBackend`, `CNNIdentityCache`, `HeadTailAnalyzer`, etc. | Drop deleted symbols; add a backwards-compat shim re-exporting the new types under their old names (one cycle), then delete |
| `core/identity/pose/__init__.py` | re-exports `YoloNativeBackend`, `SleapServiceBackend`, `auto_export_*`, `build_runtime_config`, `create_pose_backend_from_config` | Drop all deleted re-exports; keep `quality`, `artifacts`, `types` |
| `core/identity/properties/__init__.py` | re-exports `IndividualPropertiesCache`, `DetectedPropertiesCache` | Drop both; new pose cache is internal to `core/inference/cache/pose.py` |
| `data/__init__.py` | re-exports `DetectionCache` from deleted `detection_cache.py` | Drop re-export; or alias to `core/inference/cache/detection.py` for one cycle |
| `core/identity/pose/features.py` | calls `pose_props_cache.get_frame(frame_idx)` and reads `frame["detection_ids"]` / `frame["pose_keypoints"]` | Rewire to read from new `PoseCache` via a thin adapter that returns the same dict-shaped frame; keypoint map building logic unchanged |
| `core/identity/properties/export.py` | imports `CNNIdentityCache` and `DetectedPropertiesCache` | Rewire to read from new `CNNCache` and `PoseCache`; CSV column generation logic unchanged |
| `core/identity/dataset/oriented_video.py` | imports `DetectionCache` for crop export | Switch to `core/inference/cache/detection.py`; OBB read path is the only consumer here |
| `core/tracking/optimizer.py`, `optimizer_workers.py` | import `DetectionFilter` from deleted `detection_filter.py`; import `DetectionCache` from deleted `data/detection_cache.py` | `DetectionFilter` logic moves to `stages/filtering.py`; expose a shim function `apply_detection_filter(raw_obb, config) -> OBBResult` that the optimizer calls. Cache reads switch to new `DetectionCache` |
| `core/tracking/streaming_payload.py` | `StreamingAnalysisPayload` is constructed only by deleted `live_features.py` | Either: (a) construct it inside the new `runner.run_realtime()` return path so existing consumers still receive it, or (b) delete `streaming_payload.py` after confirming no consumer outside the deleted code reads it. Pick (a) to keep the GUI streaming overlay working |
| `posekit/gui/workers.py` | lazy-imports `build_runtime_config`, `create_pose_backend_from_config` from deleted `core/identity/pose/api.py` | Replace with a small public helper in `core/inference/api.py` that wraps `_load_pose_model` for single-image use; PoseKit GUI is otherwise out of scope |
| `core/tracking/worker.py` | dozens of imports from deleted modules (see "shrinks significantly" below) | Rewritten as part of the migration; covered by the feature-flag plan |

### Explicitly deleted after new pipeline is verified working

```
core/tracking/detection_phase.py
core/tracking/precompute.py
core/tracking/pose_pipeline.py
core/tracking/live_features.py
core/tracking/cnn_features.py
core/tracking/tag_features.py
core/tracking/evidence_emitter.py
core/detectors/yolo_detector.py
core/detectors/factory.py
core/detectors/detection_filter.py
core/identity/classification/cnn.py
core/identity/classification/headtail.py
core/identity/pose/api.py
core/identity/pose/backends/yolo.py
core/identity/pose/backends/sleap.py
core/identity/pose/backends/sleap_utils.py
data/detection_cache.py
data/tag_observation_cache.py
core/identity/properties/cache.py
core/identity/properties/detected_cache.py
```

### worker.py — shrinks significantly, not deleted

The ~900 lines of phase builders (lines 546–936) and inference orchestration inside the 1,377-line `run()` method are replaced by `InferenceRunner(config, cache_dir)` + `runner.run_batch_pass()` / `runner.run_realtime()`. Kalman, assignment, identity evidence building, CSV writing, GUI signals, and backward pass remain.

---

## Migration Strategy

The naive "build alongside, feature-flag, verify, delete" approach is unsafe: five files outside the rewrite scope (`optimizer.py`, `optimizer_workers.py`, `properties/export.py`, `dataset/oriented_video.py`, `posekit/gui/workers.py`) and seven `__init__.py` re-exports import from deleted modules at top level. Removing those modules without first rewiring the importers crashes on Python module load — no feature flag protects against `ImportError`.

The corrected sequence:

1. **Build `core/inference/` alongside existing code.** No existing file is modified. New module is fully unit-tested in isolation.
2. **Wire `worker.py` to use `InferenceRunner` behind a feature flag** (`USE_NEW_INFERENCE_PIPELINE`). Existing paths still work.
3. **Rewire affected kept files** (the table in "Affected kept files") *before* the feature flag flip:
   - Update each kept consumer to import from the new module (or a thin shim layer) while the deleted modules still exist.
   - Run the test suite after each rewire — both old and new code paths must still pass.
   - This step is the longest single block of work and has the most subtle bugs (cache-shape mismatches, silent metadata loss).
4. **Run output comparison** on at least one real video with the flag flipped. Verify trajectory CSVs and identity decoder outputs match within float tolerance. The new pipeline emits `cache_schema_version=2` caches; legacy `v1` caches are auto-invalidated on first run.
5. **Test migration.** Update or delete every test in `tests/` that imports from a deleted module. The full list is enumerated in the plan's `Task 18` step (5 files at minimum: `test_detection_cache.py`, `test_tag_observation_cache.py`, `test_individual_properties_cache.py`, `test_pose_pipeline.py`, `test_tag_features.py`).
6. **Update `__init__.py` re-exports.** Drop deleted symbols from `core/detectors/__init__.py`, `core/identity/classification/__init__.py`, `core/identity/pose/__init__.py`, `core/identity/properties/__init__.py`, `data/__init__.py`. This is the last step before deletion — once the re-exports are gone, the deleted files are unreachable from any kept consumer.
7. **Remove the feature flag.**
8. **Delete the files** listed in the deletion list. Run `python -m pytest tests/ -m "not benchmark"` and `python -c "import hydra_suite"` (and the same for each app entry point: `trackerkit`, `posekit`, etc.) to verify no `ImportError` regressions.
9. **Remove background-subtraction references** from `InferenceConfig` only if `bg_detector.py` is not in scope for this sprint — `bg_detector.py` itself is kept, but it must not be reached through `core/detectors/__init__.py` once the YOLO factory is gone. Either re-export `bg_detector` directly or have callers import the module path explicitly.

### What the feature flag actually protects

The flag protects only the runtime dispatch inside `worker.py` — it lets the legacy and new code paths coexist for cache verification. It does NOT protect against import-time failures in unrelated kept files. Step 3 (rewire affected kept files) is therefore mandatory even if the feature flag is left enabled, because the moment a deleted file is removed, all importers crash regardless of any runtime flag.
