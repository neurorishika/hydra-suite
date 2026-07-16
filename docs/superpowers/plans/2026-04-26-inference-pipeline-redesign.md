# Inference Pipeline Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fragmented inference pipeline with a clean `core/inference/` module that has two modes (RT/non-RT), two runtime paths (CUDA GPU-first, CPU/MPS CPU-first), a typed config schema, a producer-consumer pipeline, and per-type sidecar caches with automatic invalidation.

**Architecture:** `InferenceRunner` owns model lifecycle and exposes two methods to `worker.py`: `run_realtime(frame)` for frame-by-frame RT tracking and `run_batch_pass(video)` for the non-RT inference pass. All inference logic lives in plain stage functions under `stages/`. All persistence lives in `cache/`. `worker.py` is only modified to wire in `InferenceRunner` — Kalman, assignment, post-processing are untouched.

**Tech Stack:** Python 3.11+, PyTorch, ultralytics YOLO, OpenCV (cv2), NumPy, pytest, queue.Queue + threading.Thread for pipeline workers.

**Spec:** `docs/superpowers/specs/2026-04-26-inference-pipeline-redesign.md`

---

## Phase 1 — Foundation

Package skeleton, typed config schema, RuntimeContext, and result types. No model loading yet. All downstream tasks depend on these types being stable.

---

### Task 1: Package Skeleton + Test Infrastructure

**Files:**
- Create: `src/hydra_suite/core/inference/__init__.py`
- Create: `src/hydra_suite/core/inference/stages/__init__.py`
- Create: `src/hydra_suite/core/inference/cache/__init__.py`
- Create: `tests/test_inference_config.py`
- Create: `tests/test_inference_runtime.py`
- Create: `tests/test_inference_result.py`

- [ ] **Step 1: Create the package directories**

```bash
mkdir -p src/hydra_suite/core/inference/stages
mkdir -p src/hydra_suite/core/inference/cache
```

- [ ] **Step 2: Create empty `__init__.py` files**

`src/hydra_suite/core/inference/__init__.py`:
```python
```

`src/hydra_suite/core/inference/stages/__init__.py`:
```python
```

`src/hydra_suite/core/inference/cache/__init__.py`:
```python
```

- [ ] **Step 3: Verify Python can import the package**

```bash
python -c "import hydra_suite.core.inference; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/core/inference/
git commit -m "feat(inference): add core/inference package skeleton"
```

---

### Task 2: Config Schema

**Files:**
- Create: `src/hydra_suite/core/inference/config.py`
- Create: `tests/test_inference_config.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_inference_config.py`:
```python
import json
import pytest
import tempfile
from hydra_suite.core.inference.config import (
    InferenceConfig, InferenceConfigError,
    OBBConfig, OBBDirectConfig, OBBSequentialConfig,
    HeadTailConfig, CNNConfig,
    PoseConfig, PoseYOLOConfig,
    AprilTagConfig, CUDA_RUNTIMES, CPU_RUNTIMES,
)


def _minimal_cpu_config() -> InferenceConfig:
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/tmp/obb.pt", compute_runtime="cpu"),
        )
    )


def _minimal_cuda_config() -> InferenceConfig:
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/tmp/obb.pt", compute_runtime="cuda"),
        )
    )


def test_from_json_round_trip():
    config = _minimal_cpu_config()
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        config.to_json(f.name)
        path = f.name
    loaded = InferenceConfig.from_json(path)
    assert loaded.obb.mode == "direct"
    assert loaded.obb.direct.model_path == "/tmp/obb.pt"
    assert loaded.obb.direct.compute_runtime == "cpu"


def test_round_trip_with_headtail():
    config = InferenceConfig(
        obb=OBBConfig(mode="direct", direct=OBBDirectConfig(model_path="/m.pt")),
        headtail=HeadTailConfig(model_path="/ht.pt", compute_runtime="cpu"),
        detection_batch_size=4,
    )
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        config.to_json(f.name)
        path = f.name
    loaded = InferenceConfig.from_json(path)
    assert loaded.headtail.model_path == "/ht.pt"
    assert loaded.detection_batch_size == 4


def test_round_trip_with_cnn_phases():
    config = InferenceConfig(
        obb=OBBConfig(mode="direct", direct=OBBDirectConfig(model_path="/m.pt")),
        cnn_phases=[
            CNNConfig(label="identity", model_path="/cnn.pt", compute_runtime="cpu"),
        ],
    )
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        config.to_json(f.name)
        path = f.name
    loaded = InferenceConfig.from_json(path)
    assert len(loaded.cnn_phases) == 1
    assert loaded.cnn_phases[0].label == "identity"


def test_runtime_validation_rejects_cuda_cpu_mix():
    config = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt", compute_runtime="cuda"),
        ),
        headtail=HeadTailConfig(model_path="/ht.pt", compute_runtime="cpu"),
    )
    with pytest.raises(InferenceConfigError, match="Cannot mix"):
        config._validate_runtime_consistency()


def test_runtime_validation_accepts_cuda_group():
    config = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt", compute_runtime="cuda"),
        ),
        headtail=HeadTailConfig(model_path="/ht.pt", compute_runtime="tensorrt"),
        cnn_phases=[CNNConfig(label="id", model_path="/c.pt", compute_runtime="onnx_cuda")],
    )
    config._validate_runtime_consistency()  # must not raise


def test_runtime_validation_accepts_cpu_group():
    config = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt", compute_runtime="mps"),
        ),
        headtail=HeadTailConfig(model_path="/ht.pt", compute_runtime="onnx_coreml"),
    )
    config._validate_runtime_consistency()  # must not raise


def test_from_json_validates_on_load():
    config = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt", compute_runtime="cuda"),
        ),
        headtail=HeadTailConfig(model_path="/ht.pt", compute_runtime="cpu"),
    )
    # Bypass validation to write invalid JSON
    import json, dataclasses
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        # Write directly using to_json internals bypassing _validate
        f.write(json.dumps({"obb": {"mode": "direct",
                                     "direct": {"model_path": "/m.pt",
                                                "compute_runtime": "cuda",
                                                "confidence_floor": 0.001,
                                                "confidence_threshold": 0.25}},
                             "headtail": {"model_path": "/ht.pt",
                                          "compute_runtime": "cpu",
                                          "confidence_threshold": 0.5,
                                          "batch_size": 64,
                                          "canonical_aspect_ratio": 2.0,
                                          "canonical_margin": 1.3}}))
        path = f.name
    with pytest.raises(InferenceConfigError):
        InferenceConfig.from_json(path)


def test_sequential_config_round_trip():
    config = InferenceConfig(
        obb=OBBConfig(
            mode="sequential",
            sequential=OBBSequentialConfig(
                detect_model_path="/detect.pt",
                obb_model_path="/obb.pt",
                detect_compute_runtime="cuda",
                obb_compute_runtime="tensorrt",
                detect_confidence_threshold=0.1,
                obb_confidence_threshold=0.05,
            ),
        )
    )
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        config.to_json(f.name)
        path = f.name
    loaded = InferenceConfig.from_json(path)
    assert loaded.obb.sequential.detect_compute_runtime == "cuda"
    assert loaded.obb.sequential.obb_compute_runtime == "tensorrt"
    assert loaded.obb.sequential.detect_confidence_threshold == pytest.approx(0.1)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_inference_config.py -v 2>&1 | head -20
```
Expected: `ImportError` or `ModuleNotFoundError` — `config.py` does not exist yet.

- [ ] **Step 3: Implement `config.py`**

`src/hydra_suite/core/inference/config.py`:
```python
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Literal, Any

ComputeRuntime = Literal[
    "cpu", "mps", "cuda", "onnx_cpu", "onnx_cuda", "onnx_coreml", "tensorrt"
]

CUDA_RUNTIMES: frozenset[str] = frozenset({"cuda", "onnx_cuda", "tensorrt"})
CPU_RUNTIMES: frozenset[str] = frozenset({"cpu", "mps", "onnx_cpu", "onnx_coreml"})


class InferenceConfigError(ValueError):
    pass


@dataclass
class OBBDirectConfig:
    model_path: str
    compute_runtime: ComputeRuntime = "cpu"
    confidence_floor: float = 1e-3
    confidence_threshold: float = 0.25


@dataclass
class OBBSequentialConfig:
    detect_model_path: str
    obb_model_path: str
    detect_compute_runtime: ComputeRuntime = "cpu"
    obb_compute_runtime: ComputeRuntime = "cpu"
    detect_confidence_threshold: float = 1e-3
    obb_confidence_threshold: float = 1e-3
    detect_image_size: int = 0
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
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45


@dataclass
class HeadTailConfig:
    model_path: str
    compute_runtime: ComputeRuntime = "cpu"
    confidence_threshold: float = 0.5
    candidate_confidence_threshold: float | None = None
    batch_size: int = 64
    canonical_aspect_ratio: float = 2.0
    canonical_margin: float = 1.3


@dataclass
class CNNConfig:
    label: str
    model_path: str
    compute_runtime: ComputeRuntime = "cpu"
    confidence_threshold: float = 0.5
    batch_size: int = 64
    scoring_mode: Literal["atomic", "per_head_average"] = "atomic"
    match_bonus: float = 0.1
    mismatch_penalty: float = 0.3
    calibration_temperature: float = 1.0


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
    detection_batch_size: int = 1
    realtime: bool = False
    use_cache: bool = True
    cache_dir: str | None = None

    @staticmethod
    def from_json(path: str) -> "InferenceConfig":
        with open(path) as f:
            data = json.load(f)
        config = _dict_to_config(data)
        config._validate_runtime_consistency()
        return config

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(_config_to_dict(self), f, indent=2)

    def _collect_all_runtimes(self) -> set[str]:
        runtimes: set[str] = set()
        if self.obb.direct:
            runtimes.add(self.obb.direct.compute_runtime)
        if self.obb.sequential:
            runtimes.add(self.obb.sequential.detect_compute_runtime)
            runtimes.add(self.obb.sequential.obb_compute_runtime)
        if self.headtail:
            runtimes.add(self.headtail.compute_runtime)
        for phase in self.cnn_phases:
            runtimes.add(phase.compute_runtime)
        if self.pose:
            if self.pose.yolo:
                runtimes.add(self.pose.yolo.compute_runtime)
            if self.pose.sleap:
                runtimes.add(self.pose.sleap.compute_runtime)
        return runtimes

    def _validate_runtime_consistency(self) -> None:
        runtimes = self._collect_all_runtimes()
        uses_cuda = bool(runtimes & CUDA_RUNTIMES)
        uses_cpu = bool(runtimes & CPU_RUNTIMES)
        if uses_cuda and uses_cpu:
            raise InferenceConfigError(
                f"Cannot mix CUDA-group and CPU-group runtimes. "
                f"CUDA-group found: {runtimes & CUDA_RUNTIMES}, "
                f"CPU-group found: {runtimes & CPU_RUNTIMES}"
            )


# ── serialization helpers ─────────────────────────────────────────────────────

def _config_to_dict(config: InferenceConfig) -> dict[str, Any]:
    d = asdict(config)
    # float("inf") is not valid JSON — represent as null
    obb = d["obb"]
    if obb.get("max_object_size") == float("inf"):
        obb["max_object_size"] = None
    return d


def _dict_to_config(d: dict[str, Any]) -> InferenceConfig:
    obb_d = d["obb"]
    if obb_d.get("max_object_size") is None:
        obb_d["max_object_size"] = float("inf")

    direct = (OBBDirectConfig(**obb_d["direct"])
              if obb_d.get("direct") else None)
    sequential = (OBBSequentialConfig(**obb_d["sequential"])
                  if obb_d.get("sequential") else None)
    obb = OBBConfig(
        mode=obb_d["mode"],
        direct=direct,
        sequential=sequential,
        target_classes=obb_d.get("target_classes", []),
        max_detections=obb_d.get("max_detections", 20),
        min_object_size=obb_d.get("min_object_size", 0.0),
        max_object_size=obb_d.get("max_object_size", float("inf")),
        confidence_threshold=obb_d.get("confidence_threshold", 0.25),
        iou_threshold=obb_d.get("iou_threshold", 0.45),
    )

    ht_d = d.get("headtail")
    headtail = HeadTailConfig(**ht_d) if ht_d else None

    cnn_phases = [CNNConfig(**c) for c in d.get("cnn_phases", [])]

    pose_d = d.get("pose")
    pose = None
    if pose_d:
        yolo_d = pose_d.pop("yolo", None)
        sleap_d = pose_d.pop("sleap", None)
        bg = pose_d.get("background_color")
        if isinstance(bg, list):
            pose_d["background_color"] = tuple(bg)
        pose = PoseConfig(
            **pose_d,
            yolo=PoseYOLOConfig(**yolo_d) if yolo_d else None,
            sleap=PoseSLEAPConfig(**sleap_d) if sleap_d else None,
        )

    at_d = d.get("apriltag", {})
    if isinstance(at_d.get("unsharp_kernel"), list):
        at_d["unsharp_kernel"] = tuple(at_d["unsharp_kernel"])
    apriltag = AprilTagConfig(**at_d) if at_d else AprilTagConfig()

    return InferenceConfig(
        obb=obb,
        headtail=headtail,
        cnn_phases=cnn_phases,
        pose=pose,
        apriltag=apriltag,
        detection_batch_size=d.get("detection_batch_size", 1),
        realtime=d.get("realtime", False),
        use_cache=d.get("use_cache", True),
        cache_dir=d.get("cache_dir"),
    )
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_inference_config.py -v
```
Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/config.py tests/test_inference_config.py
git commit -m "feat(inference): add typed config schema with runtime validation"
```

---

### Task 3: RuntimeContext

**Files:**
- Create: `src/hydra_suite/core/inference/runtime.py`
- Create: `tests/test_inference_runtime.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_inference_runtime.py`:
```python
import pytest
from unittest.mock import patch
from hydra_suite.core.inference.config import (
    InferenceConfig, OBBConfig, OBBDirectConfig, HeadTailConfig,
)
from hydra_suite.core.inference.runtime import RuntimeContext


def _cpu_config() -> InferenceConfig:
    return InferenceConfig(
        obb=OBBConfig(mode="direct",
                      direct=OBBDirectConfig(model_path="/m.pt", compute_runtime="cpu"))
    )


def _mps_config() -> InferenceConfig:
    return InferenceConfig(
        obb=OBBConfig(mode="direct",
                      direct=OBBDirectConfig(model_path="/m.pt", compute_runtime="mps"))
    )


def _cuda_config() -> InferenceConfig:
    return InferenceConfig(
        obb=OBBConfig(mode="direct",
                      direct=OBBDirectConfig(model_path="/m.pt", compute_runtime="cuda"))
    )


def test_cpu_config_produces_cpu_mode():
    ctx = RuntimeContext.from_config(_cpu_config())
    assert ctx.cuda_mode is False
    assert ctx.device == "cpu"
    assert ctx.use_nvdec is False
    assert ctx.default_runtime == "cpu"


def test_mps_config_produces_cpu_mode():
    # MPS is CPU-group — cuda_mode should be False
    ctx = RuntimeContext.from_config(_mps_config())
    assert ctx.cuda_mode is False


def test_cuda_config_produces_cuda_mode():
    with patch("hydra_suite.core.inference.runtime._cuda_device_available",
               return_value="cuda:0"):
        with patch("hydra_suite.core.inference.runtime._nvdec_available",
                   return_value=True):
            ctx = RuntimeContext.from_config(_cuda_config())
    assert ctx.cuda_mode is True
    assert ctx.device == "cuda:0"
    assert ctx.use_nvdec is True
    assert ctx.default_runtime == "cuda"


def test_cuda_without_nvdec():
    with patch("hydra_suite.core.inference.runtime._cuda_device_available",
               return_value="cuda:0"):
        with patch("hydra_suite.core.inference.runtime._nvdec_available",
                   return_value=False):
            ctx = RuntimeContext.from_config(_cuda_config())
    assert ctx.cuda_mode is True
    assert ctx.use_nvdec is False


def test_frozen_dataclass():
    ctx = RuntimeContext(cuda_mode=False, device="cpu",
                         use_nvdec=False, default_runtime="cpu")
    with pytest.raises(Exception):
        ctx.cuda_mode = True  # type: ignore
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_inference_runtime.py -v 2>&1 | head -10
```
Expected: `ImportError` — `runtime.py` does not exist.

- [ ] **Step 3: Implement `runtime.py`**

`src/hydra_suite/core/inference/runtime.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

from .config import InferenceConfig, ComputeRuntime, CUDA_RUNTIMES


@dataclass(frozen=True)
class RuntimeContext:
    cuda_mode: bool
    device: str               # "cuda:0", "mps", or "cpu"
    use_nvdec: bool           # cuda_mode AND NVDEC available
    default_runtime: ComputeRuntime

    @staticmethod
    def from_config(config: InferenceConfig) -> "RuntimeContext":
        runtimes = config._collect_all_runtimes()
        cuda_mode = bool(runtimes & CUDA_RUNTIMES)
        if cuda_mode:
            device = _cuda_device_available()
            nvdec = _nvdec_available()
        else:
            device = _cpu_or_mps_device()
            nvdec = False
        default: ComputeRuntime = "cuda" if cuda_mode else "cpu"
        return RuntimeContext(
            cuda_mode=cuda_mode,
            device=device,
            use_nvdec=nvdec,
            default_runtime=default,
        )


def _cuda_device_available() -> str:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA runtime requested but no CUDA device is available. "
            "Check your CUDA installation or switch to a CPU-group runtime."
        )
    return "cuda:0"


def _cpu_or_mps_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _nvdec_available() -> bool:
    try:
        import torchvision
        return torchvision.get_video_backend() == "cuda"
    except Exception:
        return False
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_inference_runtime.py -v
```
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/runtime.py tests/test_inference_runtime.py
git commit -m "feat(inference): add RuntimeContext with CUDA/CPU path selection"
```

---

### Task 4: Result Types + Heading Resolution

**Files:**
- Create: `src/hydra_suite/core/inference/result.py`
- Create: `tests/test_inference_result.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_inference_result.py`:
```python
import math
import numpy as np
import pytest
from hydra_suite.core.inference.result import (
    OBBResult, HeadTailResult, PoseResult,
    CNNFactorPrediction, CNNDetectionPrediction, CNNResult,
    AprilTagResult, FrameResult,
    assemble_resolved_headings,
)


def _make_obb(n: int = 3) -> OBBResult:
    return OBBResult(
        frame_idx=0,
        centroids=np.zeros((n, 2)),
        angles=np.array([0.1, 0.2, 0.3][:n]),
        sizes=np.ones(n) * 500.0,
        shapes=np.ones((n, 2)),
        confidences=np.ones(n) * 0.9,
        corners=np.zeros((n, 4, 2)),
    )


def test_obb_result_num_detections():
    obb = _make_obb(3)
    assert obb.num_detections == 3


def test_resolved_headings_fallback_to_obb():
    obb = _make_obb(2)
    headings = assemble_resolved_headings(obb, None, None, None, overrides_headtail=True)
    np.testing.assert_array_almost_equal(headings, obb.angles)


def test_resolved_headings_headtail_overrides_obb():
    obb = _make_obb(2)
    headtail = HeadTailResult(
        heading_hints=np.array([1.5, float("nan")]),
        heading_confidences=np.array([0.9, 0.0]),
        directed_mask=np.array([1, 0], dtype=np.uint8),
        canonical_affines=np.zeros((2, 2, 3)),
    )
    headings = assemble_resolved_headings(obb, headtail, None, None)
    assert headings[0] == pytest.approx(1.5)   # headtail wins
    assert headings[1] == pytest.approx(0.2)   # fallback to OBB angle


def test_resolved_headings_pose_overrides_headtail():
    obb = _make_obb(2)
    headtail = HeadTailResult(
        heading_hints=np.array([1.5, 2.0]),
        heading_confidences=np.array([0.9, 0.9]),
        directed_mask=np.array([1, 1], dtype=np.uint8),
        canonical_affines=np.zeros((2, 2, 3)),
    )
    pose_headings = np.array([0.3, float("nan")])
    pose_valid = np.array([True, False])
    headings = assemble_resolved_headings(
        obb, headtail, pose_headings, pose_valid, overrides_headtail=True
    )
    assert headings[0] == pytest.approx(0.3)   # pose wins
    assert headings[1] == pytest.approx(2.0)   # fallback to headtail


def test_resolved_headings_pose_does_not_override_when_flag_false():
    obb = _make_obb(1)
    headtail = HeadTailResult(
        heading_hints=np.array([1.5]),
        heading_confidences=np.array([0.9]),
        directed_mask=np.array([1], dtype=np.uint8),
        canonical_affines=np.zeros((1, 2, 3)),
    )
    pose_headings = np.array([0.3])
    pose_valid = np.array([True])
    headings = assemble_resolved_headings(
        obb, headtail, pose_headings, pose_valid, overrides_headtail=False
    )
    assert headings[0] == pytest.approx(1.5)   # headtail wins when flag is False


def test_cnn_result_multi_head_structure():
    pred = CNNDetectionPrediction(
        det_index=0,
        factors=[
            CNNFactorPrediction("color", ["red", "blue"], np.array([0.8, 0.2])),
            CNNFactorPrediction("size", ["small", "large"], np.array([0.3, 0.7])),
        ],
    )
    assert len(pred.factors) == 2
    assert pred.factors[0].factor_name == "color"
    np.testing.assert_array_almost_equal(pred.factors[1].raw_probabilities, [0.3, 0.7])


def test_apriltag_result_empty():
    result = AprilTagResult(tag_ids=[], det_indices=[],
                            centers=np.zeros((0, 2)), corners=np.zeros((0, 4, 2)))
    assert len(result.tag_ids) == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_inference_result.py -v 2>&1 | head -10
```
Expected: `ImportError`.

- [ ] **Step 3: Implement `result.py`**

`src/hydra_suite/core/inference/result.py`:
```python
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class OBBResult:
    frame_idx: int
    centroids: np.ndarray    # (D, 2)  cx, cy
    angles: np.ndarray       # (D,)    radians
    sizes: np.ndarray        # (D,)    area px²
    shapes: np.ndarray       # (D, 2)  ellipse_area, aspect_ratio
    confidences: np.ndarray  # (D,)    raw detection confidence
    corners: np.ndarray      # (D, 4, 2) OBB corners

    @property
    def num_detections(self) -> int:
        return int(len(self.confidences))


@dataclass
class HeadTailResult:
    heading_hints: np.ndarray        # (D,) radians; nan = no confident direction
    heading_confidences: np.ndarray  # (D,)
    directed_mask: np.ndarray        # (D,) uint8; 1 = heading trusted
    canonical_affines: np.ndarray    # (D, 2, 3)


@dataclass
class CNNFactorPrediction:
    factor_name: str
    class_names: list[str]
    raw_probabilities: np.ndarray    # (num_classes,) pre-calibration


@dataclass
class CNNDetectionPrediction:
    det_index: int
    factors: list[CNNFactorPrediction]   # len=1 flat; len=K multi-head


@dataclass
class CNNResult:
    label: str                                   # from CNNConfig.label
    predictions: list[CNNDetectionPrediction]    # one per detection


@dataclass
class PoseResult:
    keypoints: np.ndarray    # (D, K, 3): [x, y, confidence] per keypoint
    valid_mask: np.ndarray   # (D,) bool: meets min_kpt_conf + min_valid_kpts


@dataclass
class AprilTagResult:
    tag_ids: list[int]
    det_indices: list[int]    # which OBB detection each tag maps to
    centers: np.ndarray       # (T, 2)
    corners: np.ndarray       # (T, 4, 2)


@dataclass
class FrameResult:
    frame_idx: int
    obb: OBBResult
    filtered_indices: list[int]        # detections that survived filtering
    headtail: HeadTailResult | None
    cnn: list[CNNResult]               # one per CNN phase
    pose: PoseResult | None
    apriltag: AprilTagResult | None
    resolved_headings: np.ndarray      # (D,) final merged heading per detection


def assemble_resolved_headings(
    obb: OBBResult,
    headtail: HeadTailResult | None,
    pose_headings: np.ndarray | None,   # (D,) nan where pose unavailable
    pose_valid: np.ndarray | None,      # (D,) bool
    overrides_headtail: bool = True,
) -> np.ndarray:
    """Merge headings with priority: pose → headtail → OBB axis.

    When overrides_headtail=False the priority is: headtail → pose → OBB axis.
    """
    result = obb.angles.copy()

    if headtail is not None:
        for i in range(obb.num_detections):
            if headtail.directed_mask[i] and not math.isnan(float(headtail.heading_hints[i])):
                result[i] = headtail.heading_hints[i]

    if pose_headings is not None and pose_valid is not None:
        for i in range(obb.num_detections):
            if not pose_valid[i]:
                continue
            if math.isnan(float(pose_headings[i])):
                continue
            if overrides_headtail:
                result[i] = pose_headings[i]
            elif headtail is None or not headtail.directed_mask[i]:
                result[i] = pose_headings[i]

    return result
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_inference_result.py -v
```
Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/result.py tests/test_inference_result.py
git commit -m "feat(inference): add result types and heading resolution logic"
```

---

*Phase 1 complete. All foundation types are stable. Proceed to Phase 2 — Stage Functions.*

---

## Phase 2 — Stage Functions

Plain functions, one file per inference type. No I/O, no mode branching, no device detection. Each file also defines the model-handle dataclass (`OBBModels`, `HeadTailModel`, etc.) used by the runner. All seven tasks can be committed independently.

---

### Task 5: OBB Stage

**Files:**
- Create: `src/hydra_suite/core/inference/stages/obb.py`
- Create: `tests/test_inference_stages_obb.py`

Ports core logic from `src/hydra_suite/core/detectors/yolo_detector.py` (direct + sequential modes) and `src/hydra_suite/core/detectors/_direct_obb_runtime.py`. On the CUDA path, `run_obb()` returns `_RawOBBTensors` objects that hold CUDA tensors without any `.cpu()` call - filtering is deferred to Task 6. Sequential mode always returns `OBBResult` (crop offset arithmetic is CPU-side). CPU/MPS path also returns `OBBResult` directly.

- [ ] **Step 1: Write failing tests**

`tests/test_inference_stages_obb.py`:
```python
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock
from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.obb import (
    OBBModels, _RawOBBTensors, run_obb,
    _extract_obb_result, _extract_raw_tensors,
    _empty_obb_result, _merge_obb_results,
)


def _cpu_rt() -> RuntimeContext:
    return RuntimeContext(cuda_mode=False, device="cpu", use_nvdec=False, default_runtime="cpu")


def _cuda_rt() -> RuntimeContext:
    return RuntimeContext(cuda_mode=True, device="cuda:0", use_nvdec=False, default_runtime="cuda")


def _mock_ul_result_tensors(n: int = 2) -> MagicMock:
    """Fake ultralytics OBB result with PyTorch tensors (CPU for testing)."""
    xywhr = torch.tensor([[100., 100., 20., 10., 0.5]] * n)
    corners = torch.zeros(n, 4, 2)
    conf = torch.full((n,), 0.8)
    r = MagicMock()
    r.obb.xywhr = xywhr
    r.obb.xyxyxyxy = corners
    r.obb.conf = conf
    r.obb.__len__ = lambda self: n
    return r


def _mock_ul_result_numpy_compat(n: int = 2) -> MagicMock:
    """Fake ultralytics OBB result via .cpu().numpy() chain (CPU-path test)."""
    def _t(arr):
        m = MagicMock()
        m.cpu.return_value.numpy.return_value = arr
        return m

    xywhr = np.array([[100., 100., 20., 10., 0.5]] * n, dtype=np.float32)
    corners = np.zeros((n, 4, 2), dtype=np.float32)
    conf = np.full(n, 0.8, dtype=np.float32)
    r = MagicMock()
    r.obb.xywhr = _t(xywhr)
    r.obb.xyxyxyxy = _t(corners)
    r.obb.conf = _t(conf)
    r.obb.__len__ = lambda self: n
    return r


def test_empty_obb_result_shape():
    r = _empty_obb_result(0)
    assert r.num_detections == 0
    assert r.centroids.shape == (0, 2)
    assert r.corners.shape == (0, 4, 2)


def test_extract_obb_result_n_detections():
    result = _extract_obb_result(_mock_ul_result_numpy_compat(n=3), frame_idx=0)
    assert result.num_detections == 3
    assert result.centroids.shape == (3, 2)
    assert result.angles.shape == (3,)
    assert result.sizes.shape == (3,)
    assert result.corners.shape == (3, 4, 2)


def test_extract_obb_result_offset_shifts_centroids():
    result = _extract_obb_result(_mock_ul_result_numpy_compat(n=1), frame_idx=0, offset=(50.0, 30.0))
    assert result.centroids[0, 0] == pytest.approx(150.0)
    assert result.centroids[0, 1] == pytest.approx(130.0)


def test_extract_obb_result_sizes_computed():
    result = _extract_obb_result(_mock_ul_result_numpy_compat(n=1), frame_idx=0)
    assert result.sizes[0] == pytest.approx(20.0 * 10.0)


def test_extract_raw_tensors_returns_named_tuple():
    r = _mock_ul_result_tensors(n=2)
    raw = _extract_raw_tensors(r, frame_idx=5)
    assert isinstance(raw, _RawOBBTensors)
    assert raw.frame_idx == 5
    assert raw.xywhr.shape == (2, 5)
    assert raw.corners.shape == (2, 4, 2)
    assert raw.conf.shape == (2,)


def test_extract_raw_tensors_no_cpu_call():
    """_extract_raw_tensors must not call .cpu() on any tensor field."""
    xywhr_mock = MagicMock(spec=torch.Tensor)
    corners_mock = MagicMock(spec=torch.Tensor)
    conf_mock = MagicMock(spec=torch.Tensor)
    r = MagicMock()
    r.obb.xywhr = xywhr_mock
    r.obb.xyxyxyxy = corners_mock
    r.obb.conf = conf_mock
    r.obb.__len__ = lambda self: 2
    _extract_raw_tensors(r, frame_idx=0)
    xywhr_mock.cpu.assert_not_called()
    corners_mock.cpu.assert_not_called()
    conf_mock.cpu.assert_not_called()


def test_merge_obb_results_concatenates():
    r1 = OBBResult(frame_idx=0, centroids=np.ones((2, 2), dtype=np.float32),
                   angles=np.ones(2, dtype=np.float32), sizes=np.ones(2, dtype=np.float32),
                   shapes=np.ones((2, 2), dtype=np.float32),
                   confidences=np.ones(2, dtype=np.float32),
                   corners=np.zeros((2, 4, 2), dtype=np.float32))
    r2 = OBBResult(frame_idx=0, centroids=np.ones((3, 2), dtype=np.float32),
                   angles=np.ones(3, dtype=np.float32), sizes=np.ones(3, dtype=np.float32),
                   shapes=np.ones((3, 2), dtype=np.float32),
                   confidences=np.ones(3, dtype=np.float32),
                   corners=np.zeros((3, 4, 2), dtype=np.float32))
    merged = _merge_obb_results(0, [r1, r2])
    assert merged.num_detections == 5


def test_run_obb_cpu_returns_obb_result():
    config = OBBConfig(mode="direct",
                       direct=OBBDirectConfig(model_path="/m.pt", compute_runtime="cpu"))
    mock_model = MagicMock()
    mock_model.predict.return_value = [_mock_ul_result_numpy_compat(n=2)]
    models = OBBModels(mode="direct", direct_model=mock_model)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = run_obb([frame], models, config, _cpu_rt())
    assert len(results) == 1
    assert isinstance(results[0], OBBResult)
    assert results[0].num_detections == 2


def test_run_obb_cuda_returns_raw_tensors():
    config = OBBConfig(mode="direct",
                       direct=OBBDirectConfig(model_path="/m.pt", compute_runtime="cuda"))
    mock_model = MagicMock()
    mock_model.predict.return_value = [_mock_ul_result_tensors(n=2)]
    models = OBBModels(mode="direct", direct_model=mock_model)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = run_obb([frame], models, config, _cuda_rt())
    assert len(results) == 1
    assert isinstance(results[0], _RawOBBTensors)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_inference_stages_obb.py -v 2>&1 | head -10
```
Expected: `ImportError` -- `stages/obb.py` does not exist.

- [ ] **Step 3: Implement `stages/obb.py`**

`src/hydra_suite/core/inference/stages/obb.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch

from ..config import OBBConfig, ComputeRuntime
from ..result import OBBResult
from ..runtime import RuntimeContext


class _RawOBBTensors(NamedTuple):
    """CUDA tensors from OBB model -- no .cpu() call until filter_from_tensors()."""
    frame_idx: int
    xywhr: torch.Tensor    # (N, 5): cx, cy, w, h, angle_rad on device
    corners: torch.Tensor  # (N, 4, 2): corner coords on device
    conf: torch.Tensor     # (N,): confidence on device


@dataclass
class OBBModels:
    mode: str                       # "direct" or "sequential"
    direct_model: Any | None = None
    detect_model: Any | None = None  # sequential stage-1
    obb_model: Any | None = None     # sequential stage-2

    def close(self) -> None:
        pass  # ultralytics models don't need explicit cleanup


def load_obb_models(config: OBBConfig, runtime: RuntimeContext) -> OBBModels:
    if config.mode == "direct":
        assert config.direct is not None
        m = _load_yolo(config.direct.model_path, config.direct.compute_runtime)
        return OBBModels(mode="direct", direct_model=m)
    assert config.sequential is not None
    detect = _load_yolo(config.sequential.detect_model_path,
                        config.sequential.detect_compute_runtime)
    obb = _load_yolo(config.sequential.obb_model_path,
                     config.sequential.obb_compute_runtime)
    return OBBModels(mode="sequential", detect_model=detect, obb_model=obb)


def run_obb(
    frames: list[np.ndarray | torch.Tensor],
    models: OBBModels,
    config: OBBConfig,
    runtime: RuntimeContext,
) -> list[OBBResult | _RawOBBTensors]:
    """Run OBB detection on a batch of frames.

    CUDA direct path: returns _RawOBBTensors per frame -- no .cpu() pull.
    CPU/MPS or sequential path: returns OBBResult per frame.
    iou=1.0 disables YOLO's internal NMS -- filtering stage handles it.
    """
    if models.mode == "direct":
        return _run_direct(frames, models.direct_model, config, runtime)
    return _run_sequential(frames, models, config, runtime)


def _run_direct(
    frames: list,
    model: Any,
    config: OBBConfig,
    runtime: RuntimeContext,
) -> list[OBBResult | _RawOBBTensors]:
    conf_floor = config.direct.confidence_floor if config.direct else 1e-3
    results = model.predict(
        frames, conf=conf_floor, iou=1.0, verbose=False, device=runtime.device,
    )
    if runtime.cuda_mode:
        return [_extract_raw_tensors(r, idx) for idx, r in enumerate(results)]
    return [_extract_obb_result(r, idx) for idx, r in enumerate(results)]


def _run_sequential(
    frames: list,
    models: OBBModels,
    config: OBBConfig,
    runtime: RuntimeContext,
) -> list[OBBResult]:
    # Sequential always returns OBBResult -- per-crop offsets are CPU arithmetic.
    seq = config.sequential
    stage1 = models.detect_model.predict(
        frames, conf=seq.detect_confidence_threshold, iou=1.0,
        verbose=False, device=runtime.device,
        imgsz=seq.detect_image_size if seq.detect_image_size > 0 else None,
    )
    results: list[OBBResult] = []
    for frame_idx, (frame, s1) in enumerate(zip(frames, stage1)):
        boxes = s1.boxes
        if boxes is None or len(boxes) == 0:
            results.append(_empty_obb_result(frame_idx))
            continue
        crops, offsets = _build_crops(frame, boxes, seq, runtime)
        if not crops:
            results.append(_empty_obb_result(frame_idx))
            continue
        batch_size = seq.stage2_batch_size or len(crops)
        sub: list[OBBResult] = []
        for i in range(0, len(crops), batch_size):
            batch = crops[i: i + batch_size]
            s2 = models.obb_model.predict(
                batch, conf=seq.obb_confidence_threshold, iou=1.0,
                verbose=False, device=runtime.device, imgsz=seq.stage2_image_size,
            )
            for j, r in enumerate(s2):
                sub.append(_extract_obb_result(r, frame_idx, offset=offsets[i + j]))
        results.append(_merge_obb_results(frame_idx, sub))
    return results


def _build_crops(
    frame: np.ndarray | torch.Tensor,
    boxes: Any,
    seq: Any,
    runtime: RuntimeContext,
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    if isinstance(frame, torch.Tensor):
        arr = frame.cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
    else:
        arr = frame
    h, w = arr.shape[:2]
    crops: list[np.ndarray] = []
    offsets: list[tuple[float, float]] = []
    for x1, y1, x2, y2 in boxes.xyxy.cpu().numpy():
        bw, bh = x2 - x1, y2 - y1
        pad = seq.crop_pad_ratio * max(bw, bh)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(bw, bh) / 2 + pad
        if seq.enforce_square_crop:
            half = max(half, seq.min_crop_size_px / 2)
        ox1 = max(0, int(cx - half))
        oy1 = max(0, int(cy - half))
        ox2 = min(w, int(cx + half))
        oy2 = min(h, int(cy + half))
        crop = arr[oy1:oy2, ox1:ox2]
        if crop.size == 0:
            continue
        crops.append(crop)
        offsets.append((float(ox1), float(oy1)))
    return crops, offsets


def _extract_raw_tensors(result: Any, frame_idx: int) -> _RawOBBTensors:
    """Keep OBB tensors on the compute device -- no .cpu() call."""
    obb = result.obb
    if obb is None or len(obb) == 0:
        device = torch.device("cuda:0")
        return _RawOBBTensors(
            frame_idx=frame_idx,
            xywhr=torch.zeros((0, 5), dtype=torch.float32, device=device),
            corners=torch.zeros((0, 4, 2), dtype=torch.float32, device=device),
            conf=torch.zeros(0, dtype=torch.float32, device=device),
        )
    return _RawOBBTensors(
        frame_idx=frame_idx,
        xywhr=obb.xywhr,
        corners=obb.xyxyxyxy,
        conf=obb.conf,
    )


def _extract_obb_result(
    result: Any,
    frame_idx: int,
    offset: tuple[float, float] = (0.0, 0.0),
) -> OBBResult:
    obb = result.obb
    if obb is None or len(obb) == 0:
        return _empty_obb_result(frame_idx)
    xywhr = obb.xywhr.cpu().numpy()       # (N, 5): cx,cy,w,h,angle
    corners = obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)
    conf = obb.conf.cpu().numpy()         # (N,)
    ox, oy = offset
    centroids = xywhr[:, :2].copy()
    centroids[:, 0] += ox
    centroids[:, 1] += oy
    corners = corners.copy()
    corners[:, :, 0] += ox
    corners[:, :, 1] += oy
    w_arr, h_arr = xywhr[:, 2], xywhr[:, 3]
    sizes = w_arr * h_arr
    aspect = np.where(h_arr > 0, w_arr / h_arr, 1.0)
    return OBBResult(
        frame_idx=frame_idx,
        centroids=centroids.astype(np.float32),
        angles=xywhr[:, 4].astype(np.float32),
        sizes=sizes.astype(np.float32),
        shapes=np.stack([sizes, aspect], axis=1).astype(np.float32),
        confidences=conf.astype(np.float32),
        corners=corners.astype(np.float32),
    )


def _empty_obb_result(frame_idx: int) -> OBBResult:
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.zeros((0, 2), dtype=np.float32),
        angles=np.zeros(0, dtype=np.float32),
        sizes=np.zeros(0, dtype=np.float32),
        shapes=np.zeros((0, 2), dtype=np.float32),
        confidences=np.zeros(0, dtype=np.float32),
        corners=np.zeros((0, 4, 2), dtype=np.float32),
    )


def _merge_obb_results(frame_idx: int, parts: list[OBBResult]) -> OBBResult:
    non_empty = [r for r in parts if r.num_detections > 0]
    if not non_empty:
        return _empty_obb_result(frame_idx)
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.concatenate([r.centroids for r in non_empty], axis=0),
        angles=np.concatenate([r.angles for r in non_empty]),
        sizes=np.concatenate([r.sizes for r in non_empty]),
        shapes=np.concatenate([r.shapes for r in non_empty], axis=0),
        confidences=np.concatenate([r.confidences for r in non_empty]),
        corners=np.concatenate([r.corners for r in non_empty], axis=0),
    )


def _load_yolo(model_path: str, compute_runtime: ComputeRuntime) -> Any:
    from ultralytics import YOLO
    device = "cuda:0" if compute_runtime in ("cuda", "onnx_cuda", "tensorrt") else (
        "mps" if compute_runtime == "mps" else "cpu"
    )
    model = YOLO(model_path)
    model.to(device)
    return model
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_inference_stages_obb.py -v
```
Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/stages/obb.py tests/test_inference_stages_obb.py
git commit -m "feat(inference): add OBB stage with GPU-aware _RawOBBTensors path"
```

---

### Task 6: Filtering Stage

**Files:**
- Create: `src/hydra_suite/core/inference/stages/filtering.py`
- Create: `tests/test_inference_stages_filtering.py`

Ports NMS + filtering logic from `src/hydra_suite/core/detectors/_obb_geometry.py` and `src/hydra_suite/core/detectors/detection_filter.py`. Two entry points:
- `filter_detections(raw: OBBResult, ...)` -- CPU/MPS path; all ops in NumPy.
- `filter_from_tensors(raw: _RawOBBTensors, ...)` -- CUDA path; confidence/size/ROI gates stay as PyTorch tensor ops on GPU; only the filtered subset's corners+xywhr+conf are pulled to CPU for NMS via `cv2.rotatedRectangleIntersection`.

Callers use `filter_raw(raw, config, roi_mask, roi_mask_cuda, runtime)` which dispatches automatically.

- [ ] **Step 1: Write failing tests**

`tests/test_inference_stages_filtering.py`:
```python
import numpy as np
import pytest
import torch
from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.obb import _RawOBBTensors, _empty_obb_result
from hydra_suite.core.inference.stages.filtering import (
    filter_detections, filter_from_tensors, filter_raw,
)


def _cpu_rt() -> RuntimeContext:
    return RuntimeContext(cuda_mode=False, device="cpu", use_nvdec=False, default_runtime="cpu")


def _cuda_rt() -> RuntimeContext:
    return RuntimeContext(cuda_mode=True, device="cuda:0", use_nvdec=False, default_runtime="cuda")


def _make_obb(centroids, confidences, sizes=None, corners=None) -> OBBResult:
    n = len(confidences)
    return OBBResult(
        frame_idx=0,
        centroids=np.array(centroids, dtype=np.float32),
        angles=np.zeros(n, dtype=np.float32),
        sizes=np.array(sizes or [500.0] * n, dtype=np.float32),
        shapes=np.ones((n, 2), dtype=np.float32),
        confidences=np.array(confidences, dtype=np.float32),
        corners=np.array(corners or [[[0, 0], [1, 0], [1, 1], [0, 1]]] * n, dtype=np.float32),
    )


def _make_raw_tensors(centroids, confidences, sizes=None) -> _RawOBBTensors:
    """Build _RawOBBTensors using CPU tensors -- no CUDA required for unit tests."""
    n = len(confidences)
    ws = [s ** 0.5 for s in (sizes or [500.0] * n)]
    xywhr = torch.tensor(
        [[centroids[i][0], centroids[i][1], ws[i], ws[i], 0.0] for i in range(n)],
        dtype=torch.float32,
    )
    corners = torch.zeros(n, 4, 2, dtype=torch.float32)
    conf = torch.tensor(confidences, dtype=torch.float32)
    return _RawOBBTensors(frame_idx=0, xywhr=xywhr, corners=corners, conf=conf)


def _cpu_config(**kwargs) -> OBBConfig:
    return OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="/m.pt"),
        **kwargs,
    )


def test_filter_confidence_gate():
    raw = _make_obb([[100, 100], [200, 200]], [0.3, 0.8])
    result = filter_detections(raw, _cpu_config(confidence_threshold=0.5))
    assert result.num_detections == 1
    assert result.confidences[0] == pytest.approx(0.8)


def test_filter_min_size_gate():
    raw = _make_obb([[100, 100], [200, 200]], [0.9, 0.9], sizes=[50.0, 500.0])
    result = filter_detections(raw, _cpu_config(min_object_size=100.0))
    assert result.num_detections == 1
    assert result.sizes[0] == pytest.approx(500.0)


def test_filter_max_size_gate():
    raw = _make_obb([[100, 100], [200, 200]], [0.9, 0.9], sizes=[50.0, 5000.0])
    result = filter_detections(raw, _cpu_config(max_object_size=1000.0))
    assert result.num_detections == 1
    assert result.sizes[0] == pytest.approx(50.0)


def test_filter_roi_mask():
    raw = _make_obb([[50, 50], [300, 300]], [0.9, 0.9])
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[0:100, 0:100] = 255
    result = filter_detections(raw, _cpu_config(), roi_mask=mask)
    assert result.num_detections == 1
    assert result.centroids[0, 0] == pytest.approx(50.0)


def test_filter_max_detections():
    raw = _make_obb([[i * 50, 0] for i in range(10)], [0.9] * 10)
    result = filter_detections(raw, _cpu_config(max_detections=3))
    assert result.num_detections == 3


def test_filter_empty_input():
    raw = _empty_obb_result(0)
    result = filter_detections(raw, _cpu_config())
    assert result.num_detections == 0


def test_filter_all_pass():
    raw = _make_obb([[100, 100], [200, 200]], [0.9, 0.8])
    result = filter_detections(raw, _cpu_config(confidence_threshold=0.5))
    assert result.num_detections == 2


def test_filter_from_tensors_confidence_gate():
    raw = _make_raw_tensors([[100, 100], [200, 200]], [0.3, 0.8])
    result = filter_from_tensors(raw, _cpu_config(confidence_threshold=0.5), None, _cuda_rt())
    assert result.num_detections == 1
    assert result.confidences[0] == pytest.approx(0.8)


def test_filter_from_tensors_min_size_gate():
    raw = _make_raw_tensors([[100, 100], [200, 200]], [0.9, 0.9], sizes=[50.0, 500.0])
    result = filter_from_tensors(raw, _cpu_config(min_object_size=100.0), None, _cuda_rt())
    assert result.num_detections == 1
    assert result.sizes[0] == pytest.approx(500.0)


def test_filter_from_tensors_max_size_gate():
    raw = _make_raw_tensors([[100, 100], [200, 200]], [0.9, 0.9], sizes=[50.0, 5000.0])
    result = filter_from_tensors(raw, _cpu_config(max_object_size=1000.0), None, _cuda_rt())
    assert result.num_detections == 1
    assert result.sizes[0] == pytest.approx(50.0)


def test_filter_from_tensors_roi_mask():
    raw = _make_raw_tensors([[50, 50], [300, 300]], [0.9, 0.9])
    mask = torch.zeros(400, 400, dtype=torch.uint8)
    mask[0:100, 0:100] = 1
    result = filter_from_tensors(raw, _cpu_config(), mask, _cuda_rt())
    assert result.num_detections == 1
    assert result.centroids[0, 0] == pytest.approx(50.0)


def test_filter_from_tensors_empty_input():
    raw = _RawOBBTensors(
        frame_idx=0,
        xywhr=torch.zeros((0, 5)),
        corners=torch.zeros((0, 4, 2)),
        conf=torch.zeros(0),
    )
    result = filter_from_tensors(raw, _cpu_config(), None, _cuda_rt())
    assert result.num_detections == 0


def test_filter_raw_dispatches_to_cpu_path():
    raw = _make_obb([[100, 100], [200, 200]], [0.3, 0.8])
    result = filter_raw(raw, _cpu_config(confidence_threshold=0.5), None, None, _cpu_rt())
    assert isinstance(result, OBBResult)
    assert result.num_detections == 1


def test_filter_raw_dispatches_to_gpu_path():
    raw = _make_raw_tensors([[100, 100], [200, 200]], [0.3, 0.8])
    result = filter_raw(raw, _cpu_config(confidence_threshold=0.5), None, None, _cuda_rt())
    assert isinstance(result, OBBResult)
    assert result.num_detections == 1
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_inference_stages_filtering.py -v 2>&1 | head -10
```
Expected: `ImportError`.

- [ ] **Step 3: Implement `stages/filtering.py`**

`src/hydra_suite/core/inference/stages/filtering.py`:
```python
from __future__ import annotations

import cv2
import numpy as np
import torch

from ..config import OBBConfig
from ..result import OBBResult
from ..runtime import RuntimeContext
from .obb import _RawOBBTensors


def filter_raw(
    raw: OBBResult | _RawOBBTensors,
    config: OBBConfig,
    roi_mask: np.ndarray | None,
    roi_mask_cuda: torch.Tensor | None,
    runtime: RuntimeContext,
) -> OBBResult:
    """Dispatcher: CUDA path uses filter_from_tensors; CPU/MPS uses filter_detections."""
    if isinstance(raw, _RawOBBTensors):
        return filter_from_tensors(raw, config, roi_mask_cuda, runtime)
    return filter_detections(raw, config, roi_mask)


def filter_detections(
    raw: OBBResult,
    config: OBBConfig,
    roi_mask: np.ndarray | None = None,
) -> OBBResult:
    """CPU/MPS path: apply confidence, size, ROI, NMS, and max-count gates in NumPy."""
    n = raw.num_detections
    if n == 0:
        return raw

    keep = np.ones(n, dtype=bool)
    keep &= raw.confidences >= config.confidence_threshold

    if config.min_object_size > 0:
        keep &= raw.sizes >= config.min_object_size
    if config.max_object_size < float("inf"):
        keep &= raw.sizes <= config.max_object_size

    if roi_mask is not None:
        h, w = roi_mask.shape[:2]
        for i in range(n):
            if not keep[i]:
                continue
            cx, cy = int(raw.centroids[i, 0]), int(raw.centroids[i, 1])
            if 0 <= cy < h and 0 <= cx < w:
                keep[i] = bool(roi_mask[cy, cx])
            else:
                keep[i] = False

    indices = np.where(keep)[0]
    if len(indices) == 0:
        return _select(raw, indices)

    if config.iou_threshold < 1.0 and len(indices) > 1:
        indices = _obb_nms(raw, indices, config.iou_threshold)

    if config.max_detections > 0 and len(indices) > config.max_detections:
        order = np.argsort(raw.confidences[indices])[::-1][: config.max_detections]
        indices = indices[order]

    return _select(raw, indices)


def filter_from_tensors(
    raw: _RawOBBTensors,
    config: OBBConfig,
    roi_mask_cuda: torch.Tensor | None,
    runtime: RuntimeContext,
) -> OBBResult:
    """CUDA path: gates run as tensor ops on device; only survivors are pulled to CPU for NMS."""
    n = raw.xywhr.shape[0]
    if n == 0:
        return _empty_obb_result(raw.frame_idx)

    # 1. Confidence gate (tensor op on device)
    keep = raw.conf >= config.confidence_threshold

    # 2. Size gates (tensor ops on device)
    w_t = raw.xywhr[:, 2]
    h_t = raw.xywhr[:, 3]
    sizes_t = w_t * h_t
    if config.min_object_size > 0:
        keep = keep & (sizes_t >= config.min_object_size)
    if config.max_object_size < float("inf"):
        keep = keep & (sizes_t <= config.max_object_size)

    # 3. ROI mask (tensor index on device)
    if roi_mask_cuda is not None:
        mask_h, mask_w = roi_mask_cuda.shape[:2]
        cx = raw.xywhr[:, 0].long().clamp(0, mask_w - 1)
        cy = raw.xywhr[:, 1].long().clamp(0, mask_h - 1)
        keep = keep & roi_mask_cuda[cy, cx].bool()

    indices_t = keep.nonzero(as_tuple=True)[0]  # still on device
    if indices_t.numel() == 0:
        return _empty_obb_result(raw.frame_idx)

    # 4. Single transfer: pull only the surviving subset to CPU
    xywhr_np = raw.xywhr[indices_t].cpu().numpy()     # (M, 5)
    corners_np = raw.corners[indices_t].cpu().numpy()  # (M, 4, 2)
    conf_np = raw.conf[indices_t].cpu().numpy()        # (M,)
    sizes_np = (xywhr_np[:, 2] * xywhr_np[:, 3]).astype(np.float32)
    aspect_np = np.where(
        xywhr_np[:, 3] > 0, xywhr_np[:, 2] / xywhr_np[:, 3], 1.0
    ).astype(np.float32)

    subset = OBBResult(
        frame_idx=raw.frame_idx,
        centroids=xywhr_np[:, :2].astype(np.float32),
        angles=xywhr_np[:, 4].astype(np.float32),
        sizes=sizes_np,
        shapes=np.stack([sizes_np, aspect_np], axis=1),
        confidences=conf_np.astype(np.float32),
        corners=corners_np.astype(np.float32),
    )

    m = subset.num_detections
    local_idx = np.arange(m)

    # 5. NMS on CPU over the already-filtered subset
    if config.iou_threshold < 1.0 and m > 1:
        local_idx = _obb_nms(subset, local_idx, config.iou_threshold)

    # 6. Cap at max_detections
    if config.max_detections > 0 and len(local_idx) > config.max_detections:
        order = np.argsort(subset.confidences[local_idx])[::-1][: config.max_detections]
        local_idx = local_idx[order]

    return _select(subset, local_idx)


def _obb_nms(raw: OBBResult, indices: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS over oriented bounding boxes using cv2.rotatedRectangleIntersection."""
    order = indices[np.argsort(raw.confidences[indices])[::-1]]
    keep: list[int] = []
    suppressed = np.zeros(len(raw.confidences), dtype=bool)

    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(int(idx))
        rect_a = _obb_to_cv2_rect(raw, idx)
        for other in order:
            if suppressed[other] or other == idx:
                continue
            if _rotated_iou(rect_a, _obb_to_cv2_rect(raw, other)) > iou_threshold:
                suppressed[other] = True

    return np.array(keep, dtype=int)


def _obb_to_cv2_rect(raw: OBBResult, idx: int) -> tuple:
    """Convert OBBResult entry to cv2 RotatedRect tuple: (center, (w, h), angle_deg)."""
    cx, cy = float(raw.centroids[idx, 0]), float(raw.centroids[idx, 1])
    corners = raw.corners[idx]  # (4, 2)
    w = float(np.linalg.norm(corners[1] - corners[0]))
    h = float(np.linalg.norm(corners[3] - corners[0]))
    angle = float(np.degrees(raw.angles[idx]))
    return (cx, cy), (w, h), angle


def _rotated_iou(rect_a: tuple, rect_b: tuple) -> float:
    """IOU between two cv2 RotatedRect tuples."""
    try:
        ret, intersection = cv2.rotatedRectangleIntersection(rect_a, rect_b)
        if ret == cv2.INTERSECT_NONE or intersection is None:
            return 0.0
        inter_area = cv2.contourArea(intersection)
        (_, (wa, ha), _) = rect_a
        (_, (wb, hb), _) = rect_b
        union = wa * ha + wb * hb - inter_area
        return float(inter_area / union) if union > 0 else 0.0
    except Exception:
        return 0.0


def _select(raw: OBBResult, indices: np.ndarray) -> OBBResult:
    if len(indices) == 0:
        return _empty_obb_result(raw.frame_idx)
    return OBBResult(
        frame_idx=raw.frame_idx,
        centroids=raw.centroids[indices],
        angles=raw.angles[indices],
        sizes=raw.sizes[indices],
        shapes=raw.shapes[indices],
        confidences=raw.confidences[indices],
        corners=raw.corners[indices],
    )


def _empty_obb_result(frame_idx: int) -> OBBResult:
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.zeros((0, 2), dtype=np.float32),
        angles=np.zeros(0, dtype=np.float32),
        sizes=np.zeros(0, dtype=np.float32),
        shapes=np.zeros((0, 2), dtype=np.float32),
        confidences=np.zeros(0, dtype=np.float32),
        corners=np.zeros((0, 4, 2), dtype=np.float32),
    )
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_inference_stages_filtering.py -v
```
Expected: all 14 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/stages/filtering.py tests/test_inference_stages_filtering.py
git commit -m "feat(inference): add GPU-aware filtering stage with filter_from_tensors"
```

---

### Task 7: Crops Stage

**Files:**
- Create: `src/hydra_suite/core/inference/stages/crops.py`
- Create: `tests/test_inference_stages_crops.py`

New code — no existing equivalent. GPU path uses `torch.nn.functional.affine_grid` + `grid_sample`. CPU path uses `cv2.warpAffine`. Both produce a `torch.Tensor` so downstream stage functions are device-agnostic.

- [ ] **Step 1: Write failing tests**

`tests/test_inference_stages_crops.py`:
```python
import numpy as np
import pytest
import torch
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.crops import (
    extract_canonical_crops, extract_aabb_crops,
)


def _cpu_rt() -> RuntimeContext:
    return RuntimeContext(cuda_mode=False, device="cpu", use_nvdec=False, default_runtime="cpu")


def _obb_result(n: int = 2, frame_h: int = 480, frame_w: int = 640) -> OBBResult:
    centroids = np.array([[320., 240.], [100., 100.]][:n], dtype=np.float32)
    # Build simple axis-aligned corners for each detection
    corners = np.zeros((n, 4, 2), dtype=np.float32)
    for i in range(n):
        cx, cy = centroids[i]
        corners[i] = [[cx-20, cy-10], [cx+20, cy-10], [cx+20, cy+10], [cx-20, cy+10]]
    return OBBResult(
        frame_idx=0, centroids=centroids, angles=np.zeros(n, dtype=np.float32),
        sizes=np.full(n, 400.0, dtype=np.float32),
        shapes=np.ones((n, 2), dtype=np.float32),
        confidences=np.ones(n, dtype=np.float32), corners=corners,
    )


def test_extract_canonical_crops_returns_tensor():
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    obb = _obb_result(n=2)
    crops = extract_canonical_crops(frame, obb, canonical_aspect_ratio=2.0,
                                    canonical_margin=1.3, runtime=_cpu_rt())
    assert isinstance(crops, torch.Tensor)
    assert crops.shape[0] == 2   # N crops
    assert crops.ndim == 4       # (N, C, H, W)


def test_extract_canonical_crops_empty_obb():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    obb = _obb_result(n=0)
    crops = extract_canonical_crops(frame, obb, canonical_aspect_ratio=2.0,
                                    canonical_margin=1.3, runtime=_cpu_rt())
    assert crops.shape[0] == 0


def test_extract_aabb_crops_returns_list():
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    obb = _obb_result(n=2)
    crops = extract_aabb_crops(frame, obb, padding=0.1)
    assert len(crops) == 2
    for crop in crops:
        assert isinstance(crop, np.ndarray)
        assert crop.ndim == 3


def test_extract_aabb_crops_empty_obb():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    obb = _obb_result(n=0)
    crops = extract_aabb_crops(frame, obb, padding=0.1)
    assert len(crops) == 0


def test_canonical_and_aabb_same_count():
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    obb = _obb_result(n=3)
    canonical = extract_canonical_crops(frame, obb, 2.0, 1.3, _cpu_rt())
    aabb = extract_aabb_crops(frame, obb, padding=0.1)
    assert canonical.shape[0] == len(aabb) == 3
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_inference_stages_crops.py -v 2>&1 | head -10
```
Expected: `ImportError`.

- [ ] **Step 3: Implement `stages/crops.py`**

`src/hydra_suite/core/inference/stages/crops.py`:
```python
from __future__ import annotations

import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..result import OBBResult
from ..runtime import RuntimeContext


def extract_canonical_crops(
    frame: np.ndarray | torch.Tensor,
    obb_result: OBBResult,
    canonical_aspect_ratio: float,
    canonical_margin: float,
    runtime: RuntimeContext,
) -> torch.Tensor:
    """Extract OBB-aligned canonical crops. Returns (N, C, H, W) tensor on runtime.device.

    GPU path: affine_grid + grid_sample on CUDA tensor (no CPU round-trip).
    CPU path: cv2.warpAffine per crop → stacked CPU tensor.
    """
    n = obb_result.num_detections
    if n == 0:
        return torch.zeros((0, 3, 64, 64), dtype=torch.float32)

    if runtime.cuda_mode:
        return _extract_canonical_gpu(frame, obb_result, canonical_aspect_ratio,
                                      canonical_margin, runtime.device)
    return _extract_canonical_cpu(frame, obb_result, canonical_aspect_ratio,
                                  canonical_margin)


def extract_aabb_crops(
    frame: np.ndarray,
    obb_result: OBBResult,
    padding: float,
) -> list[np.ndarray]:
    """Extract axis-aligned bounding box crops for AprilTag detection.
    Always CPU numpy. frame must be a numpy array (already .cpu().numpy() on CUDA path)."""
    if obb_result.num_detections == 0:
        return []
    h, w = frame.shape[:2]
    crops: list[np.ndarray] = []
    for i in range(obb_result.num_detections):
        corners = obb_result.corners[i]   # (4, 2)
        x1, y1 = corners[:, 0].min(), corners[:, 1].min()
        x2, y2 = corners[:, 0].max(), corners[:, 1].max()
        bw, bh = x2 - x1, y2 - y1
        pad = padding * max(bw, bh)
        ox1 = max(0, int(x1 - pad))
        oy1 = max(0, int(y1 - pad))
        ox2 = min(w, int(x2 + pad))
        oy2 = min(h, int(y2 + pad))
        crop = frame[oy1:oy2, ox1:ox2]
        crops.append(crop if crop.size > 0 else np.zeros((1, 1, 3), dtype=np.uint8))
    return crops


# ── CPU canonical crop extraction ─────────────────────────────────────────────

def _extract_canonical_cpu(
    frame: np.ndarray | torch.Tensor,
    obb: OBBResult,
    aspect_ratio: float,
    margin: float,
) -> torch.Tensor:
    if isinstance(frame, torch.Tensor):
        arr = frame.cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
    else:
        arr = frame

    crops: list[np.ndarray] = []
    for i in range(obb.num_detections):
        crop = _warp_canonical_crop(arr, obb.centroids[i], obb.angles[i],
                                    obb.sizes[i], aspect_ratio, margin)
        crops.append(crop)

    stacked = np.stack(crops, axis=0)           # (N, H, W, C)
    t = torch.from_numpy(stacked).permute(0, 3, 1, 2).float() / 255.0
    return t


def _warp_canonical_crop(
    frame: np.ndarray,
    centroid: np.ndarray,
    angle: float,
    size: float,
    aspect_ratio: float,
    margin: float,
) -> np.ndarray:
    """Extract a rotated crop centred on centroid, aligned so OBB is upright."""
    side = math.sqrt(size) * margin
    out_w = int(side * aspect_ratio)
    out_h = int(side)
    out_w = max(out_w, 4)
    out_h = max(out_h, 4)

    cx, cy = float(centroid[0]), float(centroid[1])
    angle_deg = float(np.degrees(angle))

    # Affine: rotate around centroid, then translate to output canvas centre
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    M[0, 2] += out_w / 2 - cx
    M[1, 2] += out_h / 2 - cy

    crop = cv2.warpAffine(frame, M, (out_w, out_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return crop


# ── GPU canonical crop extraction ─────────────────────────────────────────────

def _extract_canonical_gpu(
    frame: torch.Tensor,
    obb: OBBResult,
    aspect_ratio: float,
    margin: float,
    device: str,
) -> torch.Tensor:
    """Affine crop extraction on CUDA tensor using grid_sample."""
    if isinstance(frame, np.ndarray):
        if frame.ndim == 3:
            frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
        frame = frame.to(device)

    if frame.ndim == 3:
        frame = frame.unsqueeze(0)   # (1, C, H, W)

    _, C, H, W = frame.shape
    crops: list[torch.Tensor] = []

    for i in range(obb.num_detections):
        cx = float(obb.centroids[i, 0])
        cy = float(obb.centroids[i, 1])
        angle = float(obb.angles[i])
        side = math.sqrt(float(obb.sizes[i])) * margin
        out_w = max(int(side * aspect_ratio), 4)
        out_h = max(int(side), 4)

        # Build theta matrix for affine_grid (maps output → input normalised coords)
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)
        # Scale: output pixel → input pixel
        sx = out_w / W
        sy = out_h / H
        # Normalised centre
        ncx = 2.0 * cx / W - 1.0
        ncy = 2.0 * cy / H - 1.0

        theta = torch.tensor([
            [cos_a * sx, -sin_a * sx, ncx],
            [sin_a * sy,  cos_a * sy, ncy],
        ], dtype=torch.float32, device=device).unsqueeze(0)

        grid = F.affine_grid(theta, (1, C, out_h, out_w), align_corners=False)
        crop = F.grid_sample(frame, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=False)
        crops.append(crop.squeeze(0))   # (C, out_h, out_w)

    if not crops:
        return torch.zeros((0, C, 4, 4), device=device)

    # Pad to uniform size (max H, max W across all crops)
    max_h = max(c.shape[1] for c in crops)
    max_w = max(c.shape[2] for c in crops)
    padded = [F.pad(c, (0, max_w - c.shape[2], 0, max_h - c.shape[1])) for c in crops]
    return torch.stack(padded, dim=0)   # (N, C, H, W)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_inference_stages_crops.py -v
```
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/stages/crops.py tests/test_inference_stages_crops.py
git commit -m "feat(inference): add crops stage with GPU affine and CPU warpAffine paths"
```

---

### Task 8: HeadTail Stage

**Files:**
- Create: `src/hydra_suite/core/inference/stages/headtail.py`
- Create: `tests/test_inference_stages_headtail.py`

Uses existing `ClassifierBackend` from `src/hydra_suite/core/identity/classification/backend.py`. The label→heading mapping (right=0, left=π, up=−π/2, down=+π/2) is preserved from `src/hydra_suite/core/identity/classification/headtail.py`.

- [ ] **Step 1: Write failing tests**

`tests/test_inference_stages_headtail.py`:
```python
import math
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch
from hydra_suite.core.inference.config import HeadTailConfig
from hydra_suite.core.inference.result import OBBResult, HeadTailResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.headtail import (
    HeadTailModel, run_headtail, _label_to_heading_offset,
)


def _cpu_rt():
    return RuntimeContext(cuda_mode=False, device="cpu", use_nvdec=False, default_runtime="cpu")


def _obb(n: int = 2) -> OBBResult:
    return OBBResult(
        frame_idx=0,
        centroids=np.zeros((n, 2), dtype=np.float32),
        angles=np.zeros(n, dtype=np.float32),
        sizes=np.ones(n, dtype=np.float32) * 400.0,
        shapes=np.ones((n, 2), dtype=np.float32),
        confidences=np.ones(n, dtype=np.float32) * 0.9,
        corners=np.zeros((n, 4, 2), dtype=np.float32),
    )


def test_label_to_heading_right():
    assert _label_to_heading_offset("right") == pytest.approx(0.0)


def test_label_to_heading_left():
    assert _label_to_heading_offset("left") == pytest.approx(math.pi)


def test_label_to_heading_up():
    assert _label_to_heading_offset("up") == pytest.approx(-math.pi / 2)


def test_label_to_heading_down():
    assert _label_to_heading_offset("down") == pytest.approx(math.pi / 2)


def test_label_to_heading_unknown_returns_none():
    assert _label_to_heading_offset("unknown") is None


def test_run_headtail_empty_crops_returns_nan_hints():
    config = HeadTailConfig(model_path="/ht.pt")
    mock_backend = MagicMock()
    model = HeadTailModel(backend=mock_backend, input_size=(64, 64),
                          class_names=["right", "left", "up", "down", "unknown"])
    empty_crops = torch.zeros((0, 3, 64, 64))
    result = run_headtail(empty_crops, _obb(n=2), model, config, _cpu_rt())
    assert result.num_detections == 2
    assert all(math.isnan(h) for h in result.heading_hints)
    assert all(m == 0 for m in result.directed_mask)


def test_run_headtail_confident_prediction():
    config = HeadTailConfig(model_path="/ht.pt", confidence_threshold=0.5)
    # Mock backend returns [right=0.9, left=0.1, up=0.0, down=0.0, unknown=0.0]
    mock_backend = MagicMock()
    mock_backend.predict_batch.return_value = [
        [np.array([0.9, 0.1, 0.0, 0.0, 0.0])],   # det 0: right, conf=0.9
        [np.array([0.1, 0.8, 0.0, 0.0, 0.1])],   # det 1: left, conf=0.8
    ]
    model = HeadTailModel(backend=mock_backend, input_size=(64, 64),
                          class_names=["right", "left", "up", "down", "unknown"])
    crops = torch.zeros((2, 3, 64, 64))
    result = run_headtail(crops, _obb(n=2), model, config, _cpu_rt())
    assert result.directed_mask[0] == 1
    assert result.directed_mask[1] == 1
    assert result.heading_hints[0] == pytest.approx(0.0)    # right → offset 0
    assert result.heading_hints[1] == pytest.approx(math.pi)  # left → offset π


def test_run_headtail_below_threshold_not_directed():
    config = HeadTailConfig(model_path="/ht.pt", confidence_threshold=0.9)
    mock_backend = MagicMock()
    mock_backend.predict_batch.return_value = [
        [np.array([0.6, 0.4, 0.0, 0.0, 0.0])],  # conf=0.6 < 0.9 → not directed
    ]
    model = HeadTailModel(backend=mock_backend, input_size=(64, 64),
                          class_names=["right", "left", "up", "down", "unknown"])
    crops = torch.zeros((1, 3, 64, 64))
    result = run_headtail(crops, _obb(n=1), model, config, _cpu_rt())
    assert result.directed_mask[0] == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_inference_stages_headtail.py -v 2>&1 | head -10
```
Expected: `ImportError`.

- [ ] **Step 3: Implement `stages/headtail.py`**

`src/hydra_suite/core/inference/stages/headtail.py`:
```python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..config import HeadTailConfig
from ..result import OBBResult, HeadTailResult
from ..runtime import RuntimeContext

_DIRECTION_OFFSET: dict[str, float] = {
    "right": 0.0,
    "left": math.pi,
    "up": -math.pi / 2,
    "down": math.pi / 2,
}


@dataclass
class HeadTailModel:
    backend: Any       # ClassifierBackend instance
    input_size: tuple[int, int]   # (H, W) expected by the model
    class_names: list[str]

    def close(self) -> None:
        pass


def load_headtail_model(config: HeadTailConfig, runtime: RuntimeContext) -> HeadTailModel:
    from hydra_suite.core.identity.classification.backend import ClassifierBackend
    backend = ClassifierBackend(config.model_path, config.compute_runtime)
    meta = backend.metadata
    input_size = (meta.input_size[0], meta.input_size[1])
    return HeadTailModel(backend=backend, input_size=input_size,
                         class_names=list(meta.class_names[0]))


def run_headtail(
    crops: torch.Tensor,          # (N, C, H, W) — from extract_canonical_crops()
    obb_result: OBBResult,
    model: HeadTailModel,
    config: HeadTailConfig,
    runtime: RuntimeContext,
) -> HeadTailResult:
    """Classify head-tail orientation for each crop. No I/O, no mode branching."""
    n = obb_result.num_detections
    hints = np.full(n, float("nan"), dtype=np.float32)
    confs = np.zeros(n, dtype=np.float32)
    mask = np.zeros(n, dtype=np.uint8)
    affines = np.zeros((n, 2, 3), dtype=np.float32)

    if crops.shape[0] == 0 or n == 0:
        return HeadTailResult(hints, confs, mask, affines)

    # Resize crops to model input size
    resized = _resize_crops(crops, model.input_size)
    # Convert to list[np.ndarray] for ClassifierBackend
    np_crops = [resized[i].permute(1, 2, 0).cpu().numpy() for i in range(resized.shape[0])]

    # predict_batch returns [N_crops][K_factors] probability arrays
    all_probs = model.backend.predict_batch(np_crops)

    for i, probs_per_factor in enumerate(all_probs):
        factor_probs = probs_per_factor[0]   # flat model has exactly 1 factor
        winning_idx = int(np.argmax(factor_probs))
        winning_conf = float(factor_probs[winning_idx])
        if winning_conf < config.confidence_threshold:
            continue
        label = model.class_names[winning_idx]
        offset = _label_to_heading_offset(label)
        if offset is None:
            continue
        hints[i] = obb_result.angles[i] + offset
        confs[i] = winning_conf
        mask[i] = 1

    return HeadTailResult(hints, confs, mask, affines)


def _label_to_heading_offset(label: str) -> float | None:
    """Map direction label to angle offset relative to OBB major axis."""
    return _DIRECTION_OFFSET.get(label)


def _resize_crops(crops: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """Resize (N, C, H, W) tensor to (N, C, target_H, target_W)."""
    import torch.nn.functional as F
    th, tw = target_size
    if crops.shape[2] == th and crops.shape[3] == tw:
        return crops
    return F.interpolate(crops, size=(th, tw), mode="bilinear", align_corners=False)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_inference_stages_headtail.py -v
```
Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/stages/headtail.py tests/test_inference_stages_headtail.py
git commit -m "feat(inference): add headtail stage using ClassifierBackend"
```

---

### Task 9: CNN Stage

**Files:**
- Create: `src/hydra_suite/core/inference/stages/cnn.py`
- Create: `tests/test_inference_stages_cnn.py`

Uses `ClassifierBackend`. Flat and multi-head models both supported. Stores raw pre-calibration probabilities — temperature and scoring_mode are applied at tracking time by `IdentityEvidenceBuilder`.

- [ ] **Step 1: Write failing tests**

`tests/test_inference_stages_cnn.py`:
```python
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock
from hydra_suite.core.inference.config import CNNConfig
from hydra_suite.core.inference.result import OBBResult, CNNResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.cnn import CNNModel, run_cnn


def _cpu_rt():
    return RuntimeContext(cuda_mode=False, device="cpu", use_nvdec=False, default_runtime="cpu")


def _obb(n: int) -> OBBResult:
    return OBBResult(frame_idx=0,
                     centroids=np.zeros((n, 2), dtype=np.float32),
                     angles=np.zeros(n, dtype=np.float32),
                     sizes=np.ones(n, dtype=np.float32) * 400.0,
                     shapes=np.ones((n, 2), dtype=np.float32),
                     confidences=np.ones(n, dtype=np.float32),
                     corners=np.zeros((n, 4, 2), dtype=np.float32))


def _flat_model_backend(class_names: list[str]) -> MagicMock:
    """Mock ClassifierBackend for a flat (1-factor) model."""
    backend = MagicMock()
    backend.metadata.class_names = [class_names]
    backend.metadata.input_size = (64, 64)
    n_classes = len(class_names)
    backend.predict_batch.side_effect = lambda crops: [
        [np.ones(n_classes) / n_classes] for _ in crops
    ]
    return backend


def _multihead_model_backend(factor_classes: list[list[str]]) -> MagicMock:
    """Mock ClassifierBackend for a multi-head (K-factor) model."""
    backend = MagicMock()
    backend.metadata.class_names = factor_classes
    backend.metadata.input_size = (64, 64)
    backend.predict_batch.side_effect = lambda crops: [
        [np.ones(len(fc)) / len(fc) for fc in factor_classes] for _ in crops
    ]
    return backend


def test_run_cnn_flat_returns_one_factor_per_detection():
    config = CNNConfig(label="identity", model_path="/c.pt")
    backend = _flat_model_backend(["ant1", "ant2", "ant3"])
    model = CNNModel(backend=backend, input_size=(64, 64),
                     factor_names=["identity"],
                     factor_class_names=[["ant1", "ant2", "ant3"]])
    crops = torch.zeros((2, 3, 64, 64))
    result = run_cnn(crops, _obb(2), model, config, _cpu_rt())
    assert result.label == "identity"
    assert len(result.predictions) == 2
    for pred in result.predictions:
        assert len(pred.factors) == 1
        assert pred.factors[0].factor_name == "identity"
        assert len(pred.factors[0].class_names) == 3
        assert pred.factors[0].raw_probabilities.shape == (3,)


def test_run_cnn_multihead_returns_k_factors():
    config = CNNConfig(label="behavior", model_path="/c.pt")
    backend = _multihead_model_backend([["a", "b"], ["x", "y", "z"]])
    model = CNNModel(backend=backend, input_size=(64, 64),
                     factor_names=["color", "posture"],
                     factor_class_names=[["a", "b"], ["x", "y", "z"]])
    crops = torch.zeros((1, 3, 64, 64))
    result = run_cnn(crops, _obb(1), model, config, _cpu_rt())
    assert len(result.predictions[0].factors) == 2
    assert result.predictions[0].factors[0].factor_name == "color"
    assert result.predictions[0].factors[1].factor_name == "posture"
    assert result.predictions[0].factors[1].raw_probabilities.shape == (3,)


def test_run_cnn_empty_crops():
    config = CNNConfig(label="id", model_path="/c.pt")
    model = CNNModel(backend=MagicMock(), input_size=(64, 64),
                     factor_names=["id"], factor_class_names=[["a", "b"]])
    crops = torch.zeros((0, 3, 64, 64))
    result = run_cnn(crops, _obb(0), model, config, _cpu_rt())
    assert len(result.predictions) == 0


def test_run_cnn_raw_probabilities_not_calibrated():
    """Probabilities must be raw (pre-temperature) — calibration is tracking-time only."""
    config = CNNConfig(label="id", model_path="/c.pt", calibration_temperature=0.5)
    raw_probs = np.array([0.7, 0.2, 0.1])
    backend = MagicMock()
    backend.metadata.class_names = [["a", "b", "c"]]
    backend.metadata.input_size = (64, 64)
    backend.predict_batch.return_value = [[raw_probs.copy()]]
    model = CNNModel(backend=backend, input_size=(64, 64),
                     factor_names=["id"], factor_class_names=[["a", "b", "c"]])
    crops = torch.zeros((1, 3, 64, 64))
    result = run_cnn(crops, _obb(1), model, config, _cpu_rt())
    np.testing.assert_array_almost_equal(
        result.predictions[0].factors[0].raw_probabilities, raw_probs
    )
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_inference_stages_cnn.py -v 2>&1 | head -10
```
Expected: `ImportError`.

- [ ] **Step 3: Implement `stages/cnn.py`**

`src/hydra_suite/core/inference/stages/cnn.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..config import CNNConfig
from ..result import OBBResult, CNNResult, CNNDetectionPrediction, CNNFactorPrediction
from ..runtime import RuntimeContext


@dataclass
class CNNModel:
    backend: Any                        # ClassifierBackend
    input_size: tuple[int, int]         # (H, W)
    factor_names: list[str]             # one per factor (len=1 flat, len=K multi-head)
    factor_class_names: list[list[str]] # class names per factor

    def close(self) -> None:
        pass


def load_cnn_model(config: CNNConfig, runtime: RuntimeContext) -> CNNModel:
    from hydra_suite.core.identity.classification.backend import ClassifierBackend
    backend = ClassifierBackend(config.model_path, config.compute_runtime)
    meta = backend.metadata
    factor_names = [f"factor_{i}" for i in range(len(meta.class_names))]
    return CNNModel(
        backend=backend,
        input_size=(meta.input_size[0], meta.input_size[1]),
        factor_names=factor_names,
        factor_class_names=[list(cn) for cn in meta.class_names],
    )


def run_cnn(
    crops: torch.Tensor,     # (N, C, H, W) — from extract_canonical_crops()
    obb_result: OBBResult,
    model: CNNModel,
    config: CNNConfig,
    runtime: RuntimeContext,
) -> CNNResult:
    """Run CNN identity classifier. Returns raw pre-calibration probabilities.
    Temperature calibration and scoring_mode are applied at tracking time."""
    if crops.shape[0] == 0 or obb_result.num_detections == 0:
        return CNNResult(label=config.label, predictions=[])

    resized = _resize(crops, model.input_size)
    np_crops = [resized[i].permute(1, 2, 0).cpu().numpy() for i in range(resized.shape[0])]

    # predict_batch → [N_crops][K_factors] probability arrays
    all_probs = model.backend.predict_batch(np_crops)

    predictions: list[CNNDetectionPrediction] = []
    for det_idx, probs_per_factor in enumerate(all_probs):
        factors = [
            CNNFactorPrediction(
                factor_name=model.factor_names[k],
                class_names=model.factor_class_names[k],
                raw_probabilities=np.array(probs_per_factor[k], dtype=np.float32),
            )
            for k in range(len(probs_per_factor))
        ]
        predictions.append(CNNDetectionPrediction(det_index=det_idx, factors=factors))

    return CNNResult(label=config.label, predictions=predictions)


def _resize(crops: torch.Tensor, target: tuple[int, int]) -> torch.Tensor:
    th, tw = target
    if crops.shape[2] == th and crops.shape[3] == tw:
        return crops
    return F.interpolate(crops, size=(th, tw), mode="bilinear", align_corners=False)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_inference_stages_cnn.py -v
```
Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/stages/cnn.py tests/test_inference_stages_cnn.py
git commit -m "feat(inference): add CNN stage with flat and multi-head support"
```

---

### Task 10: Pose Stage

**Files:**
- Create: `src/hydra_suite/core/inference/stages/pose.py`
- Create: `tests/test_inference_stages_pose.py`

Wraps `YoloNativeBackend` from `src/hydra_suite/core/identity/pose/backends/yolo.py` and `SleapExportedBackend` from `sleap.py`. Backend selection is driven by `PoseConfig.backend`. Keypoints are mapped back to image coordinates by adding the crop offset.

- [ ] **Step 1: Write failing tests**

`tests/test_inference_stages_pose.py`:
```python
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock
from hydra_suite.core.inference.config import PoseConfig, PoseYOLOConfig
from hydra_suite.core.inference.result import OBBResult, PoseResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.pose import PoseModel, run_pose


def _cpu_rt():
    return RuntimeContext(cuda_mode=False, device="cpu", use_nvdec=False, default_runtime="cpu")


def _obb(n: int) -> OBBResult:
    return OBBResult(frame_idx=0,
                     centroids=np.array([[100., 100.]] * n, dtype=np.float32),
                     angles=np.zeros(n, dtype=np.float32),
                     sizes=np.ones(n, dtype=np.float32) * 400.0,
                     shapes=np.ones((n, 2), dtype=np.float32),
                     confidences=np.ones(n, dtype=np.float32),
                     corners=np.array([[[80,90],[120,90],[120,110],[80,110]]] * n,
                                      dtype=np.float32))


def _mock_pose_result(n_kpts: int = 4, conf: float = 0.8) -> MagicMock:
    """Mock ultralytics pose result with n_kpts keypoints."""
    r = MagicMock()
    kpts = np.zeros((1, n_kpts, 3), dtype=np.float32)
    kpts[0, :, 2] = conf
    r.keypoints.data.cpu.return_value.numpy.return_value = kpts
    return r


def test_run_pose_empty_crops():
    config = PoseConfig(yolo=PoseYOLOConfig(model_path="/p.pt"))
    model = PoseModel(backend=MagicMock(), n_keypoints=4, keypoint_names=["a","b","c","d"])
    crops = torch.zeros((0, 3, 64, 64))
    result = run_pose(crops, _obb(0), model, config, _cpu_rt())
    assert result.keypoints.shape == (0, 4, 3)
    assert result.valid_mask.shape == (0,)


def test_run_pose_shape():
    config = PoseConfig(yolo=PoseYOLOConfig(model_path="/p.pt",
                                             min_keypoint_confidence=0.5))
    mock_backend = MagicMock()
    mock_backend.predict_batch.return_value = [_mock_pose_result(4, conf=0.8),
                                                _mock_pose_result(4, conf=0.8)]
    model = PoseModel(backend=mock_backend, n_keypoints=4,
                      keypoint_names=["a", "b", "c", "d"])
    crops = torch.zeros((2, 3, 64, 64))
    result = run_pose(crops, _obb(2), model, config, _cpu_rt())
    assert result.keypoints.shape == (2, 4, 3)
    assert result.valid_mask.shape == (2,)


def test_run_pose_valid_mask_high_conf():
    config = PoseConfig(
        yolo=PoseYOLOConfig(model_path="/p.pt", min_keypoint_confidence=0.5),
        min_valid_keypoints=2,
    )
    mock_backend = MagicMock()
    # det 0: 4 high-conf keypoints → valid
    # det 1: 0 high-conf keypoints → invalid
    r0 = _mock_pose_result(4, conf=0.9)
    r1 = _mock_pose_result(4, conf=0.1)
    mock_backend.predict_batch.return_value = [r0, r1]
    model = PoseModel(backend=mock_backend, n_keypoints=4, keypoint_names=list("abcd"))
    crops = torch.zeros((2, 3, 64, 64))
    result = run_pose(crops, _obb(2), model, config, _cpu_rt())
    assert result.valid_mask[0] is True or result.valid_mask[0] == True
    assert result.valid_mask[1] is False or result.valid_mask[1] == False
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_inference_stages_pose.py -v 2>&1 | head -10
```
Expected: `ImportError`.

- [ ] **Step 3: Implement `stages/pose.py`**

`src/hydra_suite/core/inference/stages/pose.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..config import PoseConfig
from ..result import OBBResult, PoseResult
from ..runtime import RuntimeContext


@dataclass
class PoseModel:
    backend: Any              # YoloNativeBackend or SleapExportedBackend
    n_keypoints: int
    keypoint_names: list[str]

    def close(self) -> None:
        pass


def load_pose_model(config: PoseConfig, runtime: RuntimeContext) -> PoseModel:
    if config.backend == "yolo":
        assert config.yolo is not None
        from hydra_suite.core.identity.pose.backends.yolo import YoloNativeBackend
        device = ("cuda:0" if config.yolo.compute_runtime in ("cuda", "onnx_cuda", "tensorrt")
                  else ("mps" if config.yolo.compute_runtime == "mps" else "cpu"))
        skeleton = _load_skeleton(config.skeleton_file)
        backend = YoloNativeBackend(
            model_path=config.yolo.model_path,
            device=device,
            min_valid_conf=config.min_keypoint_confidence,
            keypoint_names=skeleton.get("keypoints"),
            conf=config.yolo.confidence_threshold,
            iou=config.yolo.iou_threshold,
            max_det=config.yolo.max_detections_per_crop,
            batch_size=config.yolo.batch_size,
        )
        n_kpts = len(skeleton.get("keypoints", []))
        return PoseModel(backend=backend, n_keypoints=n_kpts,
                         keypoint_names=skeleton.get("keypoints", []))
    else:
        assert config.sleap is not None
        from hydra_suite.core.identity.pose.backends.sleap import SleapExportedBackend
        skeleton = _load_skeleton(config.skeleton_file)
        backend = SleapExportedBackend(model_path=config.sleap.model_path)
        n_kpts = len(skeleton.get("keypoints", []))
        return PoseModel(backend=backend, n_keypoints=n_kpts,
                         keypoint_names=skeleton.get("keypoints", []))


def run_pose(
    crops: torch.Tensor,       # (N, C, H, W) — from extract_canonical_crops()
    obb_result: OBBResult,
    model: PoseModel,
    config: PoseConfig,
    runtime: RuntimeContext,
) -> PoseResult:
    """Run pose estimation on canonical crops. Returns keypoints in image coordinates."""
    n = obb_result.num_detections
    empty = PoseResult(
        keypoints=np.zeros((0, model.n_keypoints, 3), dtype=np.float32),
        valid_mask=np.zeros(0, dtype=bool),
    )
    if crops.shape[0] == 0 or n == 0:
        return empty

    np_crops = [crops[i].permute(1, 2, 0).cpu().numpy() for i in range(crops.shape[0])]
    raw_results = model.backend.predict_batch(np_crops)

    kpts_out = np.zeros((n, model.n_keypoints, 3), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    min_conf = config.min_keypoint_confidence if config.pose is None else config.min_keypoint_confidence
    # Use the config fields directly
    min_kpt_conf = config.yolo.confidence_threshold if (config.yolo and hasattr(config, 'min_keypoint_confidence')) else 0.2
    min_kpt_conf = config.min_keypoint_confidence
    min_valid = config.min_valid_keypoints

    for i, r in enumerate(raw_results):
        kpts = r.keypoints.data.cpu().numpy()  # (1, K, 3): x,y,conf per kpt
        if kpts.shape[0] == 0:
            continue
        kpts = kpts[0]  # (K, 3) — take best detection (max_det=1)
        kpts_out[i] = kpts
        n_confident = int(np.sum(kpts[:, 2] >= min_kpt_conf))
        valid[i] = n_confident >= min_valid

    return PoseResult(keypoints=kpts_out, valid_mask=valid)


def _load_skeleton(skeleton_file: str) -> dict:
    if not skeleton_file:
        return {}
    import json
    try:
        with open(skeleton_file) as f:
            return json.load(f)
    except Exception:
        return {}
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_inference_stages_pose.py -v
```
Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/stages/pose.py tests/test_inference_stages_pose.py
git commit -m "feat(inference): add pose stage wrapping YOLO and SLEAP backends"
```

---

### Task 11: AprilTag Stage

**Files:**
- Create: `src/hydra_suite/core/inference/stages/apriltag.py`
- Create: `tests/test_inference_stages_apriltag.py`

Wraps the composite-strip detection logic from `src/hydra_suite/core/identity/classification/apriltag.py`. Preprocessing (unsharp mask, contrast) is applied per crop before detection.

- [ ] **Step 1: Write failing tests**

`tests/test_inference_stages_apriltag.py`:
```python
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from hydra_suite.core.inference.config import AprilTagConfig
from hydra_suite.core.inference.result import OBBResult, AprilTagResult
from hydra_suite.core.inference.stages.apriltag import (
    AprilTagModel, run_apriltag, _preprocess_crop,
)


def _obb(n: int) -> OBBResult:
    return OBBResult(frame_idx=0,
                     centroids=np.zeros((n, 2), dtype=np.float32),
                     angles=np.zeros(n, dtype=np.float32),
                     sizes=np.ones(n, dtype=np.float32),
                     shapes=np.ones((n, 2), dtype=np.float32),
                     confidences=np.ones(n, dtype=np.float32),
                     corners=np.zeros((n, 4, 2), dtype=np.float32))


def test_run_apriltag_disabled_returns_empty():
    config = AprilTagConfig(enabled=False)
    model = AprilTagModel(detector=None, config=config)
    crops = [np.zeros((64, 64, 3), dtype=np.uint8)]
    result = run_apriltag(crops, _obb(1), model, config)
    assert len(result.tag_ids) == 0


def test_run_apriltag_empty_crops():
    config = AprilTagConfig(enabled=True)
    model = AprilTagModel(detector=MagicMock(), config=config)
    result = run_apriltag([], _obb(0), model, config)
    assert len(result.tag_ids) == 0
    assert result.centers.shape == (0, 2)


def test_preprocess_crop_returns_uint8():
    crop = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    config = AprilTagConfig(contrast_factor=1.5, unsharp_amount=1.5)
    result = _preprocess_crop(crop, config)
    assert result.dtype == np.uint8
    assert result.shape == crop.shape


def test_run_apriltag_no_detections():
    config = AprilTagConfig(enabled=True, tag_family="tag36h11")
    mock_detector = MagicMock()
    mock_detector.detect.return_value = []
    model = AprilTagModel(detector=mock_detector, config=config)
    crops = [np.zeros((64, 64, 3), dtype=np.uint8)]
    result = run_apriltag(crops, _obb(1), model, config)
    assert len(result.tag_ids) == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_inference_stages_apriltag.py -v 2>&1 | head -10
```
Expected: `ImportError`.

- [ ] **Step 3: Implement `stages/apriltag.py`**

`src/hydra_suite/core/inference/stages/apriltag.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..config import AprilTagConfig
from ..result import OBBResult, AprilTagResult


@dataclass
class AprilTagModel:
    detector: Any | None    # apriltag.Detector or None when disabled
    config: AprilTagConfig

    def close(self) -> None:
        pass


def load_apriltag_model(config: AprilTagConfig) -> AprilTagModel:
    if not config.enabled:
        return AprilTagModel(detector=None, config=config)
    try:
        import apriltag
        options = apriltag.DetectorOptions(
            families=config.tag_family,
            nthreads=config.threads,
            quad_decimate=config.decimate,
            quad_sigma=config.blur,
            refine_edges=int(config.refine_edges),
            decode_sharpening=config.decode_sharpening,
            max_hamming=config.max_hamming,
        )
        detector = apriltag.Detector(options)
    except ImportError:
        detector = None
    return AprilTagModel(detector=detector, config=config)


def run_apriltag(
    cpu_crops: list[np.ndarray],    # AABB crops, always CPU numpy
    obb_result: OBBResult,
    model: AprilTagModel,
    config: AprilTagConfig,
) -> AprilTagResult:
    """Detect AprilTags in each AABB crop. No I/O, no mode branching."""
    empty = AprilTagResult(tag_ids=[], det_indices=[],
                           centers=np.zeros((0, 2), dtype=np.float32),
                           corners=np.zeros((0, 4, 2), dtype=np.float32))

    if not config.enabled or model.detector is None or not cpu_crops:
        return empty

    tag_ids: list[int] = []
    det_indices: list[int] = []
    centers: list[np.ndarray] = []
    corners_list: list[np.ndarray] = []

    for det_idx, crop in enumerate(cpu_crops):
        if crop.size == 0:
            continue
        preprocessed = _preprocess_crop(crop, config)
        gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY) if preprocessed.ndim == 3 else preprocessed
        detections = model.detector.detect(gray)
        for det in detections:
            tag_id = int(det.tag_id)
            if config.max_tag_id is not None and tag_id > config.max_tag_id:
                continue
            tag_ids.append(tag_id)
            det_indices.append(det_idx)
            centers.append(np.array(det.center, dtype=np.float32))
            corners_list.append(np.array(det.corners, dtype=np.float32))

    if not tag_ids:
        return empty

    return AprilTagResult(
        tag_ids=tag_ids,
        det_indices=det_indices,
        centers=np.stack(centers, axis=0),
        corners=np.stack(corners_list, axis=0),
    )


def _preprocess_crop(crop: np.ndarray, config: AprilTagConfig) -> np.ndarray:
    """Apply unsharp mask and contrast enhancement before detection."""
    # Unsharp mask
    ksize = (config.unsharp_kernel[0] | 1, config.unsharp_kernel[1] | 1)  # ensure odd
    blurred = cv2.GaussianBlur(crop.astype(np.float32), ksize, config.unsharp_sigma)
    sharpened = crop.astype(np.float32) + config.unsharp_amount * (crop.astype(np.float32) - blurred)
    # Contrast
    sharpened = np.clip(sharpened * config.contrast_factor, 0, 255).astype(np.uint8)
    return sharpened
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_inference_stages_apriltag.py -v
```
Expected: all 4 tests pass.

- [ ] **Step 5: Run all Phase 2 tests together**

```bash
python -m pytest tests/test_inference_stages_obb.py tests/test_inference_stages_filtering.py \
  tests/test_inference_stages_crops.py tests/test_inference_stages_headtail.py \
  tests/test_inference_stages_cnn.py tests/test_inference_stages_pose.py \
  tests/test_inference_stages_apriltag.py -v
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/inference/stages/apriltag.py tests/test_inference_stages_apriltag.py
git commit -m "feat(inference): add apriltag stage with preprocessing and greedy detection"
```

---

*Phase 2 complete. All seven stage functions exist, are tested independently, and share no I/O or mode-branching. Proceed to Phase 3 — Cache System.*


## Phase 3 -- Cache System

Two tasks: add `materialize_tensors()` and define cache key functions (Task 12), then implement
the `CacheHandle` ABC and one concrete handle per inference type (Task 13).

**Design rule enforced here:** HeadTail/CNN/Pose run on ALL pre-filter OBB detections during the
batch pass. Filtering is re-applied at tracking time. This keeps each type's cache key independent
-- changing the OBB confidence threshold does NOT invalidate the HeadTail cache.

---

### Task 12: Cache Keys + materialize_tensors

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/obb.py`
- Create: `src/hydra_suite/core/inference/cache/keys.py`
- Create: `tests/test_inference_cache_keys.py`

`materialize_tensors()` converts a `_RawOBBTensors` (CUDA path) to a CPU-side `OBBResult` with
no gates applied -- needed before caching. Cache key functions hash only model-affecting fields;
thresholds (confidence_threshold, calibration_temperature, scoring_mode) are excluded so that
threshold edits never trigger a cache miss.

- [ ] **Step 1: Write failing tests**

`tests/test_inference_cache_keys.py`:
```python
import pytest
import torch
from hydra_suite.core.inference.config import (
    OBBConfig, OBBDirectConfig, OBBSequentialConfig,
    HeadTailConfig, CNNConfig, PoseConfig, PoseYOLOConfig, AprilTagConfig,
)
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages.obb import _RawOBBTensors, materialize_tensors
from hydra_suite.core.inference.cache.keys import (
    CacheKey,
    detection_cache_key, headtail_cache_key, cnn_cache_key,
    pose_cache_key, apriltag_cache_key,
)
import numpy as np


def _raw(n: int = 2) -> _RawOBBTensors:
    return _RawOBBTensors(
        frame_idx=3,
        xywhr=torch.tensor([[10., 20., 8., 4., 0.3]] * n),
        corners=torch.zeros(n, 4, 2),
        conf=torch.full((n,), 0.7),
    )


def _obb_direct(path="/m.pt", runtime="cpu", threshold=0.5) -> OBBConfig:
    return OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path=path, compute_runtime=runtime,
                               confidence_threshold=threshold),
    )


def _ht_config(path="/ht.pt", aspect=1.5, margin=0.1, threshold=0.4) -> HeadTailConfig:
    return HeadTailConfig(
        model_path=path, compute_runtime="cpu",
        confidence_threshold=threshold,
        canonical_aspect_ratio=aspect,
        canonical_margin=margin,
    )


def _cnn_config(path="/cnn.pt", label="id", temperature=1.0) -> CNNConfig:
    return CNNConfig(label=label, model_path=path, compute_runtime="cpu",
                     calibration_temperature=temperature)


def _pose_config(path="/pose.pt", padding=0.1) -> PoseConfig:
    return PoseConfig(
        backend="yolo",
        yolo=PoseYOLOConfig(model_path=path, compute_runtime="cpu"),
        crop_padding=padding,
    )


def _at_config(family="tag36h11", decimate=1.0, blur=0.0) -> AprilTagConfig:
    return AprilTagConfig(enabled=True, tag_family=family, decimate=decimate, blur=blur)


# ---- materialize_tensors ----

def test_materialize_tensors_shape():
    raw = _raw(n=3)
    result = materialize_tensors(raw)
    assert isinstance(result, OBBResult)
    assert result.frame_idx == 3
    assert result.num_detections == 3
    assert result.centroids.shape == (3, 2)
    assert result.corners.shape == (3, 4, 2)
    assert result.confidences.shape == (3,)


def test_materialize_tensors_values():
    raw = _raw(n=1)
    result = materialize_tensors(raw)
    assert result.centroids[0, 0] == pytest.approx(10.0)
    assert result.centroids[0, 1] == pytest.approx(20.0)
    assert result.confidences[0] == pytest.approx(0.7)
    assert result.sizes[0] == pytest.approx(8.0 * 4.0)


def test_materialize_tensors_empty():
    raw = _RawOBBTensors(
        frame_idx=0,
        xywhr=torch.zeros((0, 5)),
        corners=torch.zeros((0, 4, 2)),
        conf=torch.zeros(0),
    )
    result = materialize_tensors(raw)
    assert result.num_detections == 0


# ---- detection_cache_key ----

def test_detection_key_changes_with_model_path():
    k1 = detection_cache_key(_obb_direct(path="/a.pt"))
    k2 = detection_cache_key(_obb_direct(path="/b.pt"))
    assert k1 != k2


def test_detection_key_stable_with_threshold():
    k1 = detection_cache_key(_obb_direct(threshold=0.3))
    k2 = detection_cache_key(_obb_direct(threshold=0.8))
    assert k1.model_path == k2.model_path
    assert k1.config_hash == k2.config_hash


def test_detection_key_sequential_encodes_both_models():
    cfg = OBBConfig(
        mode="sequential",
        sequential=OBBSequentialConfig(
            detect_model_path="/det.pt",
            obb_model_path="/obb.pt",
        ),
    )
    k = detection_cache_key(cfg)
    assert "/det.pt" in k.model_path and "/obb.pt" in k.model_path


# ---- headtail_cache_key ----

def test_headtail_key_changes_with_model_path():
    k1 = headtail_cache_key(_ht_config(path="/a.pt"))
    k2 = headtail_cache_key(_ht_config(path="/b.pt"))
    assert k1 != k2


def test_headtail_key_stable_with_threshold():
    k1 = headtail_cache_key(_ht_config(threshold=0.3))
    k2 = headtail_cache_key(_ht_config(threshold=0.9))
    assert k1.model_path == k2.model_path
    assert k1.config_hash == k2.config_hash


def test_headtail_key_changes_with_canonical_params():
    k1 = headtail_cache_key(_ht_config(aspect=1.5, margin=0.1))
    k2 = headtail_cache_key(_ht_config(aspect=2.0, margin=0.1))
    assert k1.config_hash != k2.config_hash


# ---- cnn_cache_key ----

def test_cnn_key_stable_with_calibration_temperature():
    k1 = cnn_cache_key(_cnn_config(temperature=1.0))
    k2 = cnn_cache_key(_cnn_config(temperature=2.5))
    assert k1.model_path == k2.model_path
    assert k1.config_hash == k2.config_hash


def test_cnn_key_changes_with_model_path():
    k1 = cnn_cache_key(_cnn_config(path="/a.pt"))
    k2 = cnn_cache_key(_cnn_config(path="/b.pt"))
    assert k1 != k2


# ---- pose_cache_key ----

def test_pose_key_changes_with_crop_padding():
    k1 = pose_cache_key(_pose_config(padding=0.1))
    k2 = pose_cache_key(_pose_config(padding=0.3))
    assert k1.config_hash != k2.config_hash


# ---- apriltag_cache_key ----

def test_apriltag_key_changes_with_family():
    k1 = apriltag_cache_key(_at_config(family="tag36h11"))
    k2 = apriltag_cache_key(_at_config(family="tag25h9"))
    assert k1.config_hash != k2.config_hash


def test_apriltag_key_has_empty_model_path():
    k = apriltag_cache_key(_at_config())
    assert k.model_path == ""
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_inference_cache_keys.py -v 2>&1 | head -10
```
Expected: `ImportError` -- `stages/obb.py` has no `materialize_tensors` and `cache/keys.py` does not exist.

- [ ] **Step 3.1: Add `materialize_tensors` to `stages/obb.py`**

Append to `src/hydra_suite/core/inference/stages/obb.py` (after `_merge_obb_results`):
```python
def materialize_tensors(raw: _RawOBBTensors) -> OBBResult:
    """Pull all device tensors to CPU as OBBResult with no filtering gates applied."""
    if raw.xywhr.shape[0] == 0:
        return _empty_obb_result(raw.frame_idx)
    xywhr_np = raw.xywhr.cpu().numpy()
    corners_np = raw.corners.cpu().numpy()
    conf_np = raw.conf.cpu().numpy()
    w, h = xywhr_np[:, 2], xywhr_np[:, 3]
    sizes = (w * h).astype(np.float32)
    aspect = np.where(h > 0, w / h, 1.0).astype(np.float32)
    return OBBResult(
        frame_idx=raw.frame_idx,
        centroids=xywhr_np[:, :2].astype(np.float32),
        angles=xywhr_np[:, 4].astype(np.float32),
        sizes=sizes,
        shapes=np.stack([sizes, aspect], axis=1),
        confidences=conf_np.astype(np.float32),
        corners=corners_np.astype(np.float32),
    )
```

- [ ] **Step 3.2: Implement `cache/keys.py`**

`src/hydra_suite/core/inference/cache/keys.py`:
```python
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from ..config import (
    OBBConfig, HeadTailConfig, CNNConfig, PoseConfig, AprilTagConfig,
)


@dataclass(frozen=True)
class CacheKey:
    model_path: str    # primary model path (or "|"-joined for sequential)
    model_mtime: float  # os.path.getmtime of primary model; 0.0 if no model file
    config_hash: str   # sha256 hex of model-affecting config fields; "" when none apply

    def as_string(self) -> str:
        return f"{self.model_path}|{self.model_mtime:.6f}|{self.config_hash}"


def detection_cache_key(config: OBBConfig) -> CacheKey:
    if config.mode == "direct":
        assert config.direct is not None
        path = config.direct.model_path
    else:
        assert config.sequential is not None
        path = f"{config.sequential.detect_model_path}|{config.sequential.obb_model_path}"
    return CacheKey(
        model_path=path,
        model_mtime=_mtime(path.split("|")[0]),
        config_hash="",  # confidence_threshold excluded
    )


def headtail_cache_key(config: HeadTailConfig) -> CacheKey:
    config_hash = _sha(
        f"{config.canonical_aspect_ratio}|{config.canonical_margin}"
    )
    return CacheKey(
        model_path=config.model_path,
        model_mtime=_mtime(config.model_path),
        config_hash=config_hash,
    )


def cnn_cache_key(config: CNNConfig) -> CacheKey:
    return CacheKey(
        model_path=config.model_path,
        model_mtime=_mtime(config.model_path),
        config_hash="",  # calibration_temperature, scoring_mode excluded
    )


def pose_cache_key(config: PoseConfig) -> CacheKey:
    if config.backend == "yolo":
        assert config.yolo is not None
        path = config.yolo.model_path
    else:
        assert config.sleap is not None
        path = config.sleap.model_path
    config_hash = _sha(
        f"{config.crop_padding}|{config.suppress_foreign_regions}|{config.background_color}"
    )
    return CacheKey(
        model_path=path,
        model_mtime=_mtime(path),
        config_hash=config_hash,
    )


def apriltag_cache_key(config: AprilTagConfig) -> CacheKey:
    config_hash = _sha(
        f"{config.tag_family}|{config.decimate}|{config.blur}|{config.refine_edges}"
        f"|{config.unsharp_kernel}|{config.unsharp_sigma}|{config.unsharp_amount}"
        f"|{config.contrast_factor}"
    )
    return CacheKey(model_path="", model_mtime=0.0, config_hash=config_hash)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_inference_cache_keys.py -v
```
Expected: all 16 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/stages/obb.py \
        src/hydra_suite/core/inference/cache/keys.py \
        tests/test_inference_cache_keys.py
git commit -m "feat(inference): add cache keys and materialize_tensors for OBB tensors"
```

---

### Task 13: Cache Handles

**Files:**
- Create: `src/hydra_suite/core/inference/cache/store.py`
- Create: `tests/test_inference_cache_store.py`

One `CacheHandle` subclass per inference type. Each handle buffers written frames in memory and
flushes to a `.npz` file on `close()`. Results are stored as stacked numpy arrays with a
`frame_indices` column; no Python object serialization is used. JSON is used for string metadata
(factor names, class names) stored as numpy unicode arrays.

HeadTail/CNN/Pose handles store results indexed by pre-filter OBB `det_index` so that OBB
threshold changes never invalidate downstream caches.

- [ ] **Step 1: Write failing tests**

`tests/test_inference_cache_store.py`:
```python
import json
import numpy as np
import pytest
from pathlib import Path
from hydra_suite.core.inference.cache.keys import CacheKey
from hydra_suite.core.inference.cache.store import (
    DetectionCacheHandle, HeadTailCacheHandle, CNNCacheHandle,
    PoseCacheHandle, AprilTagCacheHandle,
)
from hydra_suite.core.inference.result import (
    OBBResult, AprilTagResult,
    CNNDetectionPrediction, CNNFactorPrediction,
)


def _key(path="/m.pt") -> CacheKey:
    return CacheKey(model_path=path, model_mtime=0.0, config_hash="abc")


def _obb(frame_idx: int, n: int = 2) -> OBBResult:
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.ones((n, 2), dtype=np.float32) * frame_idx,
        angles=np.zeros(n, dtype=np.float32),
        sizes=np.full(n, 100.0, dtype=np.float32),
        shapes=np.ones((n, 2), dtype=np.float32),
        confidences=np.full(n, 0.9, dtype=np.float32),
        corners=np.zeros((n, 4, 2), dtype=np.float32),
    )


def _cnn_preds(n_dets: int = 2) -> list[CNNDetectionPrediction]:
    return [
        CNNDetectionPrediction(
            det_index=i,
            factors=[
                CNNFactorPrediction(
                    factor_name="color",
                    class_names=["red", "blue"],
                    raw_probabilities=np.array([0.7, 0.3], dtype=np.float32),
                ),
                CNNFactorPrediction(
                    factor_name="size",
                    class_names=["small", "medium", "large"],
                    raw_probabilities=np.array([0.1, 0.6, 0.3], dtype=np.float32),
                ),
            ],
        )
        for i in range(n_dets)
    ]


# ---- DetectionCacheHandle ----

def test_detection_round_trip(tmp_path):
    path = tmp_path / "test.obb.npz"
    key = _key()
    handle = DetectionCacheHandle(path=path, key=key)
    assert not handle.is_valid()

    handle.write_frame(0, result=_obb(0, n=2))
    handle.write_frame(1, result=_obb(1, n=3))
    handle.write_frame(2, result=_obb(2, n=0))
    handle.close()

    handle2 = DetectionCacheHandle(path=path, key=key)
    assert handle2.is_valid()
    r0 = handle2.read_frame(0)
    assert r0.num_detections == 2
    assert r0.centroids[0, 0] == pytest.approx(0.0)
    r1 = handle2.read_frame(1)
    assert r1.num_detections == 3
    r2 = handle2.read_frame(2)
    assert r2.num_detections == 0


def test_detection_key_mismatch_returns_invalid(tmp_path):
    path = tmp_path / "test.obb.npz"
    handle = DetectionCacheHandle(path=path, key=_key("/a.pt"))
    handle.write_frame(0, result=_obb(0))
    handle.close()

    assert not DetectionCacheHandle(path=path, key=_key("/b.pt")).is_valid()


def test_detection_missing_file_is_invalid(tmp_path):
    assert not DetectionCacheHandle(path=tmp_path / "no.npz", key=_key()).is_valid()


# ---- HeadTailCacheHandle ----

def test_headtail_round_trip(tmp_path):
    path = tmp_path / "test.ht.npz"
    key = _key()
    handle = HeadTailCacheHandle(path=path, key=key)

    hints = np.array([0.0, 1.5], dtype=np.float32)
    confs = np.array([0.8, 0.9], dtype=np.float32)
    directed = np.array([1, 1], dtype=np.uint8)
    handle.write_frame(0, det_indices=np.array([0, 1]),
                       heading_hints=hints, heading_confidences=confs, directed_mask=directed)
    handle.close()

    handle2 = HeadTailCacheHandle(path=path, key=key)
    assert handle2.is_valid()
    h, c, d = handle2.read_frame(0)
    assert h.shape == (2,)
    assert h[1] == pytest.approx(1.5)
    assert d[0] == 1


def test_headtail_read_invalid_returns_none(tmp_path):
    assert HeadTailCacheHandle(path=tmp_path / "no.npz", key=_key()).read_frame(0) is None


# ---- CNNCacheHandle ----

def test_cnn_round_trip(tmp_path):
    path = tmp_path / "test.cnn.npz"
    key = _key()
    handle = CNNCacheHandle(path=path, key=key, label="id")

    handle.write_frame(0, predictions=_cnn_preds(n_dets=2))
    handle.close()

    handle2 = CNNCacheHandle(path=path, key=key, label="id")
    assert handle2.is_valid()
    loaded = handle2.read_frame(0)
    assert len(loaded) == 2
    det0 = loaded[0]
    assert det0.det_index == 0
    assert len(det0.factors) == 2
    assert det0.factors[0].factor_name == "color"
    assert det0.factors[0].raw_probabilities[0] == pytest.approx(0.7)
    assert det0.factors[1].factor_name == "size"
    assert det0.factors[1].raw_probabilities[1] == pytest.approx(0.6)


def test_cnn_empty_frame_round_trip(tmp_path):
    path = tmp_path / "test.cnn.npz"
    key = _key()
    handle = CNNCacheHandle(path=path, key=key, label="id")
    handle.write_frame(0, predictions=[])
    handle.close()

    handle2 = CNNCacheHandle(path=path, key=key, label="id")
    assert handle2.read_frame(0) == []


# ---- PoseCacheHandle ----

def test_pose_round_trip(tmp_path):
    path = tmp_path / "test.pose.npz"
    key = _key()
    handle = PoseCacheHandle(path=path, key=key)

    kp = np.random.rand(2, 17, 3).astype(np.float32)
    valid = np.array([True, False], dtype=bool)
    handle.write_frame(0, det_indices=np.array([0, 1]), keypoints=kp, valid_mask=valid)
    handle.close()

    handle2 = PoseCacheHandle(path=path, key=key)
    assert handle2.is_valid()
    kp2, det_idx2, valid2 = handle2.read_frame(0)
    assert kp2.shape == (2, 17, 3)
    assert kp2[0, 0, 0] == pytest.approx(kp[0, 0, 0])
    assert valid2[0] == True
    assert valid2[1] == False


# ---- AprilTagCacheHandle ----

def test_apriltag_round_trip(tmp_path):
    path = tmp_path / "test.at.npz"
    key = _key()
    handle = AprilTagCacheHandle(path=path, key=key)

    result = AprilTagResult(
        tag_ids=np.array([3, 7], dtype=np.int32),
        det_indices=np.array([0, 1], dtype=np.int32),
        centers=np.array([[10., 20.], [30., 40.]], dtype=np.float32),
        corners=np.zeros((2, 4, 2), dtype=np.float32),
    )
    handle.write_frame(0, result=result)
    handle.close()

    handle2 = AprilTagCacheHandle(path=path, key=key)
    assert handle2.is_valid()
    r2 = handle2.read_frame(0)
    assert list(r2.tag_ids) == [3, 7]
    assert r2.centers[1, 0] == pytest.approx(30.0)


def test_apriltag_empty_frame(tmp_path):
    path = tmp_path / "test.at.npz"
    key = _key()
    handle = AprilTagCacheHandle(path=path, key=key)
    empty = AprilTagResult(
        tag_ids=np.array([], dtype=np.int32),
        det_indices=np.array([], dtype=np.int32),
        centers=np.zeros((0, 2), dtype=np.float32),
        corners=np.zeros((0, 4, 2), dtype=np.float32),
    )
    handle.write_frame(0, result=empty)
    handle.close()

    r2 = AprilTagCacheHandle(path=path, key=key).read_frame(0)
    assert len(r2.tag_ids) == 0


# ---- independence ----

def test_detection_and_headtail_caches_are_independent(tmp_path):
    det_path = tmp_path / "test.obb.npz"
    ht_path = tmp_path / "test.ht.npz"

    det_handle = DetectionCacheHandle(path=det_path, key=_key("/obb.pt"))
    det_handle.write_frame(0, result=_obb(0))
    det_handle.close()

    ht_handle = HeadTailCacheHandle(path=ht_path, key=_key("/ht.pt"))
    assert not ht_handle.is_valid()
    assert DetectionCacheHandle(path=det_path, key=_key("/obb.pt")).is_valid()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_inference_cache_store.py -v 2>&1 | head -10
```
Expected: `ImportError`.

- [ ] **Step 3: Implement `cache/store.py`**

`src/hydra_suite/core/inference/cache/store.py`:
```python
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .keys import CacheKey
from ..result import (
    OBBResult, AprilTagResult,
    CNNDetectionPrediction, CNNFactorPrediction,
)


class CacheHandle(ABC):
    @abstractmethod
    def is_valid(self) -> bool: ...

    @abstractmethod
    def write_frame(self, frame_idx: int, **kwargs) -> None: ...

    @abstractmethod
    def read_frame(self, frame_idx: int) -> Any: ...

    @abstractmethod
    def close(self) -> None: ...


def _check_key(path: Path, key: CacheKey) -> bool:
    if not path.exists():
        return False
    try:
        data = np.load(path)
        return str(data["cache_key"][0]) == key.as_string()
    except Exception:
        return False


def _npz_save(path: Path, key: CacheKey, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, cache_key=np.array([key.as_string()]), **arrays)


# ---- DetectionCacheHandle ────────────────────────────────────────────────────

@dataclass
class DetectionCacheHandle(CacheHandle):
    path: Path
    key: CacheKey
    _buffer: list[OBBResult] = field(default_factory=list, repr=False)
    _data: dict | None = field(default=None, repr=False)
    _valid: bool | None = field(default=None, repr=False)

    def is_valid(self) -> bool:
        if self._valid is None:
            self._valid = _check_key(self.path, self.key)
        return self._valid

    def write_frame(self, frame_idx: int, *, result: OBBResult, **_) -> None:
        self._buffer.append(result)

    def read_frame(self, frame_idx: int) -> OBBResult | None:
        if not self.is_valid():
            return None
        if self._data is None:
            self._data = dict(np.load(self.path))
        d = self._data
        mask = d["frame_indices"] == frame_idx
        return OBBResult(
            frame_idx=frame_idx,
            centroids=d["centroids"][mask],
            angles=d["angles"][mask],
            sizes=d["sizes"][mask],
            shapes=d["shapes"][mask],
            confidences=d["confidences"][mask],
            corners=d["corners"][mask],
        )

    def close(self) -> None:
        if not self._buffer:
            return
        fi_list, cents, angs, szs, shps, confs, corns = [], [], [], [], [], [], []
        for r in self._buffer:
            n = r.num_detections
            fi_list.extend([r.frame_idx] * n)
            if n > 0:
                cents.append(r.centroids)
                angs.append(r.angles)
                szs.append(r.sizes)
                shps.append(r.shapes)
                confs.append(r.confidences)
                corns.append(r.corners)
        _npz_save(
            self.path, self.key,
            frame_count=np.array([len(self._buffer)]),
            frame_indices=np.array(fi_list, dtype=np.int32),
            centroids=np.concatenate(cents) if cents else np.zeros((0, 2), np.float32),
            angles=np.concatenate(angs) if angs else np.zeros(0, np.float32),
            sizes=np.concatenate(szs) if szs else np.zeros(0, np.float32),
            shapes=np.concatenate(shps) if shps else np.zeros((0, 2), np.float32),
            confidences=np.concatenate(confs) if confs else np.zeros(0, np.float32),
            corners=np.concatenate(corns) if corns else np.zeros((0, 4, 2), np.float32),
        )


# ---- HeadTailCacheHandle ─────────────────────────────────────────────────────

@dataclass
class HeadTailCacheHandle(CacheHandle):
    path: Path
    key: CacheKey
    _buf_fi: list[int] = field(default_factory=list, repr=False)
    _buf_det: list[np.ndarray] = field(default_factory=list, repr=False)
    _buf_hints: list[np.ndarray] = field(default_factory=list, repr=False)
    _buf_confs: list[np.ndarray] = field(default_factory=list, repr=False)
    _buf_dir: list[np.ndarray] = field(default_factory=list, repr=False)
    _data: dict | None = field(default=None, repr=False)
    _valid: bool | None = field(default=None, repr=False)

    def is_valid(self) -> bool:
        if self._valid is None:
            self._valid = _check_key(self.path, self.key)
        return self._valid

    def write_frame(self, frame_idx: int, *, det_indices: np.ndarray,
                    heading_hints: np.ndarray, heading_confidences: np.ndarray,
                    directed_mask: np.ndarray, **_) -> None:
        n = len(det_indices)
        self._buf_fi.extend([frame_idx] * n)
        self._buf_det.append(det_indices)
        self._buf_hints.append(heading_hints)
        self._buf_confs.append(heading_confidences)
        self._buf_dir.append(directed_mask)

    def read_frame(self, frame_idx: int):
        if not self.is_valid():
            return None
        if self._data is None:
            self._data = dict(np.load(self.path))
        d = self._data
        mask = d["frame_indices"] == frame_idx
        return d["heading_hints"][mask], d["heading_confidences"][mask], d["directed_mask"][mask]

    def close(self) -> None:
        _npz_save(
            self.path, self.key,
            frame_indices=np.array(self._buf_fi, dtype=np.int32),
            det_indices=np.concatenate(self._buf_det) if self._buf_det else np.zeros(0, np.int32),
            heading_hints=np.concatenate(self._buf_hints) if self._buf_hints else np.zeros(0, np.float32),
            heading_confidences=np.concatenate(self._buf_confs) if self._buf_confs else np.zeros(0, np.float32),
            directed_mask=np.concatenate(self._buf_dir) if self._buf_dir else np.zeros(0, np.uint8),
        )


# ---- CNNCacheHandle ──────────────────────────────────────────────────────────

@dataclass
class CNNCacheHandle(CacheHandle):
    path: Path
    key: CacheKey
    label: str
    _buf_fi: list[int] = field(default_factory=list, repr=False)
    _buf_det: list[int] = field(default_factory=list, repr=False)
    _buf_probs: list[np.ndarray] = field(default_factory=list, repr=False)
    _factor_names: list[str] | None = field(default=None, repr=False)
    _class_names: list[list[str]] | None = field(default=None, repr=False)
    _data: dict | None = field(default=None, repr=False)
    _valid: bool | None = field(default=None, repr=False)

    def is_valid(self) -> bool:
        if self._valid is None:
            self._valid = _check_key(self.path, self.key)
        return self._valid

    def write_frame(self, frame_idx: int, *, predictions: list[CNNDetectionPrediction], **_) -> None:
        for pred in predictions:
            if self._factor_names is None and pred.factors:
                self._factor_names = [f.factor_name for f in pred.factors]
                self._class_names = [f.class_names for f in pred.factors]
            self._buf_fi.append(frame_idx)
            self._buf_det.append(pred.det_index)
            if pred.factors:
                row = np.stack([f.raw_probabilities for f in pred.factors])  # (F, C_i)
                self._buf_probs.append(row)
            else:
                self._buf_probs.append(np.zeros((0, 0), dtype=np.float32))

    def read_frame(self, frame_idx: int) -> list[CNNDetectionPrediction] | None:
        if not self.is_valid():
            return None
        if self._data is None:
            self._data = dict(np.load(self.path))
        d = self._data
        fi = d["frame_indices"]
        mask = fi == frame_idx
        if not mask.any():
            return []
        factor_names = json.loads(str(d["factor_names_json"][0]))
        class_names_list = json.loads(str(d["class_names_json"][0]))
        class_counts = d["class_counts"].astype(int)
        probs_all = d["probabilities"]   # (M, F, C_max)
        det_indices = d["det_indices"][mask]
        probs_frame = probs_all[mask]    # (K, F, C_max)
        results = []
        for k, det_idx in enumerate(det_indices):
            factors = [
                CNNFactorPrediction(
                    factor_name=factor_names[f],
                    class_names=class_names_list[f],
                    raw_probabilities=probs_frame[k, f, :class_counts[f]].copy(),
                )
                for f in range(len(factor_names))
            ]
            results.append(CNNDetectionPrediction(det_index=int(det_idx), factors=factors))
        return results

    def close(self) -> None:
        if not self._buf_probs or self._factor_names is None:
            _npz_save(
                self.path, self.key,
                frame_indices=np.zeros(0, np.int32),
                det_indices=np.zeros(0, np.int32),
                factor_names_json=np.array([json.dumps([])]),
                class_names_json=np.array([json.dumps([])]),
                class_counts=np.zeros(0, np.int32),
                probabilities=np.zeros((0, 0, 0), np.float32),
            )
            return
        class_counts = np.array([len(cn) for cn in self._class_names], dtype=np.int32)
        c_max = int(class_counts.max())
        f_count = len(self._factor_names)
        probs_stack = np.full((len(self._buf_probs), f_count, c_max), np.nan, dtype=np.float32)
        for m, probs in enumerate(self._buf_probs):
            if probs.size == 0:
                continue
            for f_idx in range(min(f_count, probs.shape[0])):
                n_cls = probs.shape[1] if probs.ndim > 1 else 0
                probs_stack[m, f_idx, :n_cls] = probs[f_idx, :n_cls]
        _npz_save(
            self.path, self.key,
            frame_indices=np.array(self._buf_fi, dtype=np.int32),
            det_indices=np.array(self._buf_det, dtype=np.int32),
            factor_names_json=np.array([json.dumps(self._factor_names)]),
            class_names_json=np.array([json.dumps(self._class_names)]),
            class_counts=class_counts,
            probabilities=probs_stack,
        )


# ---- PoseCacheHandle ─────────────────────────────────────────────────────────

@dataclass
class PoseCacheHandle(CacheHandle):
    path: Path
    key: CacheKey
    _buf_fi: list[int] = field(default_factory=list, repr=False)
    _buf_det: list[np.ndarray] = field(default_factory=list, repr=False)
    _buf_kp: list[np.ndarray] = field(default_factory=list, repr=False)
    _buf_valid: list[np.ndarray] = field(default_factory=list, repr=False)
    _data: dict | None = field(default=None, repr=False)
    _valid: bool | None = field(default=None, repr=False)

    def is_valid(self) -> bool:
        if self._valid is None:
            self._valid = _check_key(self.path, self.key)
        return self._valid

    def write_frame(self, frame_idx: int, *, det_indices: np.ndarray,
                    keypoints: np.ndarray, valid_mask: np.ndarray, **_) -> None:
        n = len(det_indices)
        self._buf_fi.extend([frame_idx] * n)
        self._buf_det.append(det_indices)
        self._buf_kp.append(keypoints)
        self._buf_valid.append(valid_mask.astype(np.uint8))

    def read_frame(self, frame_idx: int):
        if not self.is_valid():
            return None
        if self._data is None:
            self._data = dict(np.load(self.path))
        d = self._data
        mask = d["frame_indices"] == frame_idx
        return d["keypoints"][mask], d["det_indices"][mask], d["valid_mask"][mask].astype(bool)

    def close(self) -> None:
        _npz_save(
            self.path, self.key,
            frame_indices=np.array(self._buf_fi, dtype=np.int32),
            det_indices=np.concatenate(self._buf_det) if self._buf_det else np.zeros(0, np.int32),
            keypoints=np.concatenate(self._buf_kp) if self._buf_kp else np.zeros((0, 0, 3), np.float32),
            valid_mask=np.concatenate(self._buf_valid) if self._buf_valid else np.zeros(0, np.uint8),
        )


# ---- AprilTagCacheHandle ─────────────────────────────────────────────────────

@dataclass
class AprilTagCacheHandle(CacheHandle):
    path: Path
    key: CacheKey
    _buf_fi: list[int] = field(default_factory=list, repr=False)
    _buf_tag_ids: list[np.ndarray] = field(default_factory=list, repr=False)
    _buf_det: list[np.ndarray] = field(default_factory=list, repr=False)
    _buf_centers: list[np.ndarray] = field(default_factory=list, repr=False)
    _buf_corners: list[np.ndarray] = field(default_factory=list, repr=False)
    _data: dict | None = field(default=None, repr=False)
    _valid: bool | None = field(default=None, repr=False)

    def is_valid(self) -> bool:
        if self._valid is None:
            self._valid = _check_key(self.path, self.key)
        return self._valid

    def write_frame(self, frame_idx: int, *, result: AprilTagResult, **_) -> None:
        t = len(result.tag_ids)
        self._buf_fi.extend([frame_idx] * t)
        self._buf_tag_ids.append(result.tag_ids)
        self._buf_det.append(result.det_indices)
        self._buf_centers.append(result.centers)
        self._buf_corners.append(result.corners)

    def read_frame(self, frame_idx: int) -> AprilTagResult | None:
        if not self.is_valid():
            return None
        if self._data is None:
            self._data = dict(np.load(self.path))
        d = self._data
        mask = d["frame_indices"] == frame_idx
        return AprilTagResult(
            tag_ids=d["tag_ids"][mask],
            det_indices=d["det_indices"][mask],
            centers=d["centers"][mask],
            corners=d["corners"][mask],
        )

    def close(self) -> None:
        _npz_save(
            self.path, self.key,
            frame_indices=np.array(self._buf_fi, dtype=np.int32),
            tag_ids=np.concatenate(self._buf_tag_ids) if self._buf_tag_ids else np.zeros(0, np.int32),
            det_indices=np.concatenate(self._buf_det) if self._buf_det else np.zeros(0, np.int32),
            centers=np.concatenate(self._buf_centers) if self._buf_centers else np.zeros((0, 2), np.float32),
            corners=np.concatenate(self._buf_corners) if self._buf_corners else np.zeros((0, 4, 2), np.float32),
        )
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_inference_cache_store.py -v
```
Expected: all 13 tests pass.

- [ ] **Step 5: Run all Phase 3 tests**

```bash
python -m pytest tests/test_inference_cache_keys.py tests/test_inference_cache_store.py -v
```
Expected: all 29 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/inference/cache/store.py tests/test_inference_cache_store.py
git commit -m "feat(inference): add cache handles for all five inference types"
```

---

*Phase 3 complete. Cache keys and handles are defined and tested. Each type's cache is independent -- changing one model only re-runs that type. Proceed to Phase 4 -- InferenceRunner.*

---

## Phase 4: InferenceRunner

**Goal:** Wire the five stage functions and five cache handles into a single `InferenceRunner` class that manages model lifecycle, real-time single-frame inference, and offline batch-pass video inference with per-type disk caching.

### Task 14: InferenceRunner foundation — model loading and real-time mode

**Files:**
- Modify: `src/hydra_suite/core/inference/filtering.py` — add `filter_with_indices`
- Modify: `src/hydra_suite/core/inference/cache/store.py` — fix `HeadTailCacheHandle.read_frame` to prepend `det_indices`
- Create: `src/hydra_suite/core/inference/runner.py`
- Modify: `src/hydra_suite/core/inference/__init__.py`
- Test: `tests/test_inference_runner_rt.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_inference_runner_rt.py
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from hydra_suite.core.inference.result_types import (
    OBBResult, HeadTailResult, CNNResult, PoseResult, AprilTagResult, FrameResult,
)
from hydra_suite.core.inference.config import OBBConfig, InferenceConfig


def _make_obb(n: int = 5, conf_values: list | None = None) -> OBBResult:
    rng = np.random.default_rng(42)
    conf = np.array(conf_values, dtype=np.float32) if conf_values is not None else rng.uniform(0.2, 1.0, n).astype(np.float32)
    n_actual = len(conf)
    return OBBResult(
        frame_idx=0,
        centers=rng.uniform(0, 640, (n_actual, 2)).astype(np.float32),
        xywhr=rng.uniform(10, 100, (n_actual, 5)).astype(np.float32),
        corners=rng.uniform(0, 640, (n_actual, 4, 2)).astype(np.float32),
        conf=conf,
    )


def test_filter_with_indices_returns_correct_indices():
    from hydra_suite.core.inference.filtering import filter_with_indices
    raw = _make_obb(conf_values=[0.9, 0.1, 0.8, 0.2, 0.7])
    cfg = OBBConfig(confidence_threshold=0.5, min_object_size=0, nms_iou_threshold=1.0)
    filtered, indices = filter_with_indices(raw, cfg)
    assert set(indices.tolist()) == {0, 2, 4}
    assert len(filtered.centers) == 3


def test_filter_with_indices_centers_match_original():
    from hydra_suite.core.inference.filtering import filter_with_indices
    raw = _make_obb(conf_values=[0.9, 0.1, 0.8, 0.2, 0.7])
    cfg = OBBConfig(confidence_threshold=0.5, min_object_size=0, nms_iou_threshold=1.0)
    filtered, indices = filter_with_indices(raw, cfg)
    for i, orig_idx in enumerate(indices):
        np.testing.assert_allclose(filtered.centers[i], raw.centers[orig_idx])
        np.testing.assert_allclose(filtered.conf[i], raw.conf[orig_idx])


def test_filter_with_indices_empty_result():
    from hydra_suite.core.inference.filtering import filter_with_indices
    raw = _make_obb(conf_values=[0.1, 0.2, 0.1])
    cfg = OBBConfig(confidence_threshold=0.5, min_object_size=0, nms_iou_threshold=1.0)
    filtered, indices = filter_with_indices(raw, cfg)
    assert len(indices) == 0
    assert len(filtered.centers) == 0
    assert indices.dtype == np.int32


def test_inference_runner_init_loads_models():
    from hydra_suite.core.inference.runner import InferenceRunner
    cfg = InferenceConfig(compute_runtime="cpu")
    mock_models = MagicMock()
    with patch("hydra_suite.core.inference.runner._load_all_models", return_value=mock_models) as mock_load:
        runner = InferenceRunner(cfg)
    mock_load.assert_called_once_with(cfg, runner.runtime)
    assert runner._models is mock_models


def test_inference_runner_caches_all_valid_returns_false_when_no_cache_dir():
    from hydra_suite.core.inference.runner import InferenceRunner
    cfg = InferenceConfig(compute_runtime="cpu")
    with patch("hydra_suite.core.inference.runner._load_all_models"):
        runner = InferenceRunner(cfg, cache_dir=None)
    assert runner.caches_all_valid() is False


def test_inference_runner_close_calls_model_close():
    from hydra_suite.core.inference.runner import InferenceRunner, _AllModels
    cfg = InferenceConfig(compute_runtime="cpu")
    mock_obb = MagicMock()
    mock_ht = MagicMock()
    mock_models = _AllModels(obb=mock_obb, headtail=mock_ht, cnn=[], pose=None, apriltag=None)
    with patch("hydra_suite.core.inference.runner._load_all_models", return_value=mock_models):
        runner = InferenceRunner(cfg)
    runner.close()
    mock_obb.close.assert_called_once()
    mock_ht.close.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_inference_runner_rt.py -v`
Expected: FAIL — `filter_with_indices` not yet in `filtering.py`; `InferenceRunner` not yet in `runner.py`.

- [ ] **Step 3: Add `filter_with_indices` to `filtering.py`**

`_select` and `_apply_nms` already exist in this file from Task 6. Add after `filter_detections`:

```python
def filter_with_indices(
    raw: OBBResult,
    config: OBBConfig,
    roi_mask: np.ndarray | None = None,
) -> tuple[OBBResult, np.ndarray]:
    """Run the same gates as filter_detections and return (filtered_result, pre-filter indices).

    Returned indices index into *raw*. They are stored alongside downstream
    results in the cache so that a threshold edit never invalidates those caches --
    only the OBB detection cache stores pre-filter results; all downstream caches
    are keyed by these indices and re-matched on load_frame.
    """
    keep = raw.conf >= config.confidence_threshold
    if config.min_object_size > 0:
        areas = raw.xywhr[:, 2] * raw.xywhr[:, 3]
        keep = keep & (areas >= config.min_object_size)
    if roi_mask is not None:
        h, w = roi_mask.shape[:2]
        cx = np.clip(raw.centers[:, 0].astype(np.int32), 0, w - 1)
        cy = np.clip(raw.centers[:, 1].astype(np.int32), 0, h - 1)
        keep = keep & roi_mask[cy, cx].astype(bool)
    indices = np.where(keep)[0]
    subset = _select(raw, indices)
    if config.nms_iou_threshold < 1.0 and len(indices) > 1:
        keep_nms = _apply_nms(subset, config.nms_iou_threshold)
        indices = indices[keep_nms]
        subset = _select(raw, indices)
    return subset, indices.astype(np.int32)
```

- [ ] **Step 4: Fix `HeadTailCacheHandle.read_frame` in `cache/store.py`**

The current implementation omits `det_indices` from the return tuple. Find `read_frame` inside `HeadTailCacheHandle` and replace it:

```python
    def read_frame(self, frame_idx: int):
        d = self._data
        if d is None:
            return None
        mask = d["frame_indices"] == frame_idx
        if not mask.any():
            return None
        return (
            d["det_indices"][mask].astype(np.int32),
            d["heading_hints"][mask],
            d["heading_confidences"][mask],
            d["directed_mask"][mask],
        )
```

- [ ] **Step 5: Create `runner.py`**

```python
# src/hydra_suite/core/inference/runner.py
from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .cache.keys import (
    detection_cache_key,
    headtail_cache_key,
    cnn_cache_key,
    pose_cache_key,
    apriltag_cache_key,
)
from .cache.store import (
    CacheHandle,
    DetectionCacheHandle,
    HeadTailCacheHandle,
    CNNCacheHandle,
    PoseCacheHandle,
    AprilTagCacheHandle,
)
from .config import InferenceConfig
from .filtering import filter_raw, filter_with_indices
from .result_types import (
    OBBResult,
    HeadTailResult,
    CNNResult,
    PoseResult,
    AprilTagResult,
    FrameResult,
)
from .runtime_context import RuntimeContext
from .stages.obb import run_obb, OBBModels, materialize_tensors, _RawOBBTensors
from .stages.crops import extract_canonical_crops, extract_aabb_crops
from .stages.headtail import run_headtail, HeadTailModel
from .stages.cnn import run_cnn, CNNModel
from .stages.pose import run_pose, PoseModel
from .stages.apriltag import run_apriltag, AprilTagModel


@dataclass
class _AllModels:
    obb: OBBModels
    headtail: HeadTailModel | None
    cnn: list[CNNModel]
    pose: PoseModel | None
    apriltag: AprilTagModel | None


@dataclass
class _CacheSet:
    detection: DetectionCacheHandle | None
    headtail: HeadTailCacheHandle | None
    cnn: list[CNNCacheHandle] = field(default_factory=list)
    pose: PoseCacheHandle | None = None
    apriltag: AprilTagCacheHandle | None = None

    def all_handles(self) -> list[CacheHandle]:
        handles: list[CacheHandle] = []
        if self.detection is not None:
            handles.append(self.detection)
        if self.headtail is not None:
            handles.append(self.headtail)
        handles.extend(self.cnn)
        if self.pose is not None:
            handles.append(self.pose)
        if self.apriltag is not None:
            handles.append(self.apriltag)
        return handles


def _load_all_models(config: InferenceConfig, runtime: RuntimeContext) -> _AllModels:
    from .stages.obb import load_obb_models
    from .stages.headtail import load_headtail_model
    from .stages.cnn import load_cnn_model
    from .stages.pose import load_pose_model
    from .stages.apriltag import load_apriltag_model

    obb = load_obb_models(config.obb, runtime)
    headtail = load_headtail_model(config.headtail, runtime) if config.headtail.enabled else None
    cnn = [load_cnn_model(c, runtime) for c in config.cnn if c.enabled]
    pose = load_pose_model(config.pose, runtime) if config.pose.enabled else None
    apriltag = load_apriltag_model(config.apriltag) if config.apriltag.enabled else None
    return _AllModels(obb=obb, headtail=headtail, cnn=cnn, pose=pose, apriltag=apriltag)


def _open_caches(config: InferenceConfig, cache_dir: Path) -> _CacheSet:
    enabled_cnn = [c for c in config.cnn if c.enabled]
    return _CacheSet(
        detection=DetectionCacheHandle(
            cache_dir / "detection.npz",
            detection_cache_key(config.obb),
        ),
        headtail=HeadTailCacheHandle(
            cache_dir / "headtail.npz",
            headtail_cache_key(config.headtail),
        ) if config.headtail.enabled else None,
        cnn=[
            CNNCacheHandle(cache_dir / f"cnn_{c.label}.npz", cnn_cache_key(c))
            for c in enabled_cnn
        ],
        pose=PoseCacheHandle(
            cache_dir / "pose.npz",
            pose_cache_key(config.pose),
        ) if config.pose.enabled else None,
        apriltag=AprilTagCacheHandle(
            cache_dir / "apriltag.npz",
            apriltag_cache_key(config.apriltag),
        ) if config.apriltag.enabled else None,
    )


def _build_frame_result(
    frame_idx: int,
    filtered_obb: OBBResult,
    ht_result: HeadTailResult | None,
    cnn_results: list[CNNResult],
    pose_result: PoseResult | None,
    apriltag_result: AprilTagResult | None,
) -> FrameResult:
    return FrameResult(
        frame_idx=frame_idx,
        obb=filtered_obb,
        headtail=ht_result,
        cnn=cnn_results,
        pose=pose_result,
        apriltag=apriltag_result,
    )


def _load_headtail_for_indices(
    cache: HeadTailCacheHandle | None,
    frame_idx: int,
    det_indices: np.ndarray,
    filtered_obb: OBBResult,
) -> HeadTailResult | None:
    if cache is None or len(det_indices) == 0:
        return None
    data = cache.read_frame(frame_idx)
    if data is None:
        return None
    cached_det_indices, heading_hints, heading_confs, directed_mask = data
    idx_map = {int(v): i for i, v in enumerate(cached_det_indices)}
    n = len(det_indices)
    out_hints = np.zeros(n, dtype=np.float32)
    out_confs = np.zeros(n, dtype=np.float32)
    out_directed = np.zeros(n, dtype=bool)
    for i, di in enumerate(det_indices):
        j = idx_map.get(int(di))
        if j is not None:
            out_hints[i] = heading_hints[j]
            out_confs[i] = heading_confs[j]
            out_directed[i] = directed_mask[j]
    return HeadTailResult(
        frame_idx=frame_idx,
        heading_hints=out_hints,
        heading_confidences=out_confs,
        directed_mask=out_directed,
    )


def _load_cnn_for_indices(
    caches: list[CNNCacheHandle],
    frame_idx: int,
    det_indices: np.ndarray,
    filtered_obb: OBBResult,
) -> list[CNNResult]:
    results: list[CNNResult] = []
    for cache in caches:
        data = cache.read_frame(frame_idx)
        if data is None:
            results.append(None)
            continue
        cached_det_indices, probs, factor_names, class_names = data
        idx_map = {int(v): i for i, v in enumerate(cached_det_indices)}
        n = len(det_indices)
        n_factors = probs.shape[1] if probs.ndim >= 2 else 1
        c_max = probs.shape[2] if probs.ndim >= 3 else probs.shape[-1]
        out_probs = np.zeros((n, n_factors, c_max), dtype=np.float32)
        for i, di in enumerate(det_indices):
            j = idx_map.get(int(di))
            if j is not None:
                out_probs[i] = probs[j]
        results.append(CNNResult(
            frame_idx=frame_idx,
            probabilities=out_probs,
            factor_names=factor_names,
            class_names=class_names,
        ))
    return results


def _load_pose_for_indices(
    cache: PoseCacheHandle | None,
    frame_idx: int,
    det_indices: np.ndarray,
    filtered_obb: OBBResult,
) -> PoseResult | None:
    if cache is None or len(det_indices) == 0:
        return None
    data = cache.read_frame(frame_idx)
    if data is None:
        return None
    cached_det_indices, keypoints, scores = data
    idx_map = {int(v): i for i, v in enumerate(cached_det_indices)}
    n = len(det_indices)
    kp_shape = keypoints.shape[1:] if keypoints.ndim >= 2 else (1, 2)
    score_shape = scores.shape[1:] if scores.ndim >= 2 else (scores.shape[-1],)
    out_kp = np.zeros((n, *kp_shape), dtype=np.float32)
    out_scores = np.zeros((n, *score_shape), dtype=np.float32)
    for i, di in enumerate(det_indices):
        j = idx_map.get(int(di))
        if j is not None:
            out_kp[i] = keypoints[j]
            out_scores[i] = scores[j]
    return PoseResult(frame_idx=frame_idx, keypoints=out_kp, scores=out_scores)


def _load_apriltag(
    cache: AprilTagCacheHandle | None,
    frame_idx: int,
) -> AprilTagResult | None:
    if cache is None:
        return None
    return cache.read_frame(frame_idx)


class InferenceRunner:
    def __init__(self, config: InferenceConfig, cache_dir: Path | None = None) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self.runtime = RuntimeContext.from_config(config)
        self._models = _load_all_models(config, self.runtime)
        self._caches: _CacheSet | None = None

    def caches_all_valid(self) -> bool:
        """Return True only when every enabled cache file exists and matches its key."""
        if self.cache_dir is None:
            return False
        caches = _open_caches(self.config, self.cache_dir)
        return all(h.is_valid() for h in caches.all_handles())

    def run_realtime(
        self,
        frame: np.ndarray,
        roi_mask: np.ndarray | None = None,
        roi_mask_cuda: Any = None,
    ) -> FrameResult:
        """Run full inference on a single frame. No cache I/O."""
        raw_list = run_obb([frame], self.config.obb, self._models.obb, self.runtime)
        raw = raw_list[0]
        if isinstance(raw, _RawOBBTensors):
            filtered_obb = filter_raw(raw, self.config.obb, roi_mask, roi_mask_cuda, self.runtime)
        else:
            filtered_obb, _ = filter_with_indices(raw, self.config.obb, roi_mask)

        if len(filtered_obb.centers) == 0:
            return _build_frame_result(0, filtered_obb, None, [], None, None)

        canonical_crops = extract_canonical_crops(frame, filtered_obb, self.config)
        aabb_crops = (
            extract_aabb_crops(frame, filtered_obb, self.config)
            if self._models.apriltag else []
        )
        enabled_cnn_cfgs = [c for c in self.config.cnn if c.enabled]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            ht_fut = (
                pool.submit(run_headtail, canonical_crops, self.config.headtail, self._models.headtail, self.runtime)
                if self._models.headtail else None
            )
            cnn_futs = [
                pool.submit(run_cnn, canonical_crops, cfg, mdl, self.runtime)
                for cfg, mdl in zip(enabled_cnn_cfgs, self._models.cnn)
            ]
            pose_fut = (
                pool.submit(run_pose, canonical_crops, self.config.pose, self._models.pose, self.runtime)
                if self._models.pose else None
            )
            at_fut = (
                pool.submit(run_apriltag, aabb_crops, self.config.apriltag, self._models.apriltag)
                if self._models.apriltag else None
            )
            ht_result = ht_fut.result() if ht_fut else None
            cnn_results = [f.result() for f in cnn_futs]
            pose_result = pose_fut.result() if pose_fut else None
            at_result = at_fut.result() if at_fut else None

        return _build_frame_result(0, filtered_obb, ht_result, cnn_results, pose_result, at_result)

    def run_batch_pass(self, video_path: Path, progress_cb=None) -> None:
        """Run inference on every frame of `video_path` and write results to cache.

        Requires `cache_dir` to be set. Reads frames in batches, runs OBB detection,
        materializes all tensors for storage, then runs downstream models on the
        filtered subset. Closes all cache handles on completion or failure.
        """
        import cv2

        if self.cache_dir is None:
            raise RuntimeError("cache_dir must be set before calling run_batch_pass")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        caches = _open_caches(self.config, self.cache_dir)
        self._caches = caches
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        batch_size = getattr(self.config.obb, "batch_size", 8)
        enabled_cnn_cfgs = [c for c in self.config.cnn if c.enabled]

        frames_buf: list[np.ndarray] = []
        indices_buf: list[int] = []
        processed = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_buf.append(frame)
                indices_buf.append(processed)
                processed += 1
                if len(frames_buf) == batch_size:
                    self._run_batch(frames_buf, indices_buf, caches, enabled_cnn_cfgs)
                    frames_buf.clear()
                    indices_buf.clear()
                if progress_cb and total_frames > 0 and processed % max(1, total_frames // 100) == 0:
                    progress_cb(processed, total_frames)
            if frames_buf:
                self._run_batch(frames_buf, indices_buf, caches, enabled_cnn_cfgs)
            if progress_cb:
                progress_cb(processed, total_frames)
        finally:
            cap.release()
            for h in caches.all_handles():
                h.close()

    def _run_batch(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
        caches: _CacheSet,
        enabled_cnn_cfgs: list,
    ) -> None:
        raw_list = run_obb(frames, self.config.obb, self._models.obb, self.runtime)

        for frame, frame_idx, raw in zip(frames, frame_indices, raw_list):
            # Materialize all tensors for cache storage -- no confidence gate applied here.
            # The cache stores all raw detections; filter_with_indices is applied on load_frame.
            if isinstance(raw, _RawOBBTensors):
                obb_result = materialize_tensors(raw)
            else:
                obb_result = raw

            caches.detection.write_frame(frame_idx, obb=obb_result)

            filtered_obb, det_indices = filter_with_indices(obb_result, self.config.obb)
            if len(filtered_obb.centers) == 0:
                continue

            canonical_crops = extract_canonical_crops(frame, filtered_obb, self.config)

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                ht_fut = (
                    pool.submit(run_headtail, canonical_crops, self.config.headtail, self._models.headtail, self.runtime)
                    if self._models.headtail else None
                )
                cnn_futs = [
                    pool.submit(run_cnn, canonical_crops, cfg, mdl, self.runtime)
                    for cfg, mdl in zip(enabled_cnn_cfgs, self._models.cnn)
                ]
                pose_fut = (
                    pool.submit(run_pose, canonical_crops, self.config.pose, self._models.pose, self.runtime)
                    if self._models.pose else None
                )
                ht_result = ht_fut.result() if ht_fut else None
                cnn_results = [f.result() for f in cnn_futs]
                pose_result = pose_fut.result() if pose_fut else None

            if caches.headtail and ht_result is not None:
                caches.headtail.write_frame(frame_idx, det_indices=det_indices, headtail=ht_result)
            for cache, result in zip(caches.cnn, cnn_results):
                if result is not None:
                    cache.write_frame(frame_idx, det_indices=det_indices, cnn=result)
            if caches.pose and pose_result is not None:
                caches.pose.write_frame(frame_idx, det_indices=det_indices, pose=pose_result)

            if self._models.apriltag:
                aabb_crops = extract_aabb_crops(frame, filtered_obb, self.config)
                at_result = run_apriltag(aabb_crops, self.config.apriltag, self._models.apriltag)
                if caches.apriltag and at_result is not None:
                    caches.apriltag.write_frame(frame_idx, apriltag=at_result)

    def load_frame(self, frame_idx: int) -> FrameResult:
        """Load cached inference results for `frame_idx`, re-applying OBB filtering.

        The detection cache stores pre-filter detections. filter_with_indices is
        applied here so that threshold edits take effect without re-running inference.
        Downstream results are aligned to post-filter survivors via det_indices.
        """
        if self.cache_dir is None:
            raise RuntimeError("cache_dir not set -- cannot load cached frames")
        if self._caches is None:
            self._caches = _open_caches(self.config, self.cache_dir)

        raw_obb = self._caches.detection.read_frame(frame_idx)
        if raw_obb is None:
            raise KeyError(f"Frame {frame_idx} not found in detection cache")

        filtered_obb, det_indices = filter_with_indices(raw_obb, self.config.obb)

        ht_result = _load_headtail_for_indices(self._caches.headtail, frame_idx, det_indices, filtered_obb)
        cnn_results = _load_cnn_for_indices(self._caches.cnn, frame_idx, det_indices, filtered_obb)
        pose_result = _load_pose_for_indices(self._caches.pose, frame_idx, det_indices, filtered_obb)
        at_result = _load_apriltag(self._caches.apriltag, frame_idx)

        return _build_frame_result(frame_idx, filtered_obb, ht_result, cnn_results, pose_result, at_result)

    def close(self) -> None:
        """Release all model resources."""
        self._models.obb.close()
        if self._models.headtail:
            self._models.headtail.close()
        for mdl in self._models.cnn:
            mdl.close()
        if self._models.pose:
            self._models.pose.close()
        if self._models.apriltag:
            self._models.apriltag.close()
```

- [ ] **Step 6: Update `__init__.py`**

Add to `src/hydra_suite/core/inference/__init__.py`:

```python
from .filtering import filter_with_indices
from .runner import InferenceRunner, _AllModels, _CacheSet
```

- [ ] **Step 7: Run all tests to verify they pass**

Run: `python -m pytest tests/test_inference_runner_rt.py -v`
Expected: all 6 tests pass.

- [ ] **Step 8: Run all Phase 1-4a tests**

Run: `python -m pytest tests/test_inference_config.py tests/test_inference_runtime_context.py tests/test_inference_result_types.py tests/test_inference_obb_stage.py tests/test_inference_filtering.py tests/test_inference_cache_keys.py tests/test_inference_cache_store.py tests/test_inference_runner_rt.py -v`
Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/hydra_suite/core/inference/filtering.py \
        src/hydra_suite/core/inference/cache/store.py \
        src/hydra_suite/core/inference/runner.py \
        src/hydra_suite/core/inference/__init__.py \
        tests/test_inference_runner_rt.py
git commit -m "feat(inference): add InferenceRunner with real-time inference mode and filter_with_indices"
```

---

### Task 15: InferenceRunner — batch pass and load_frame

**Files:**
- Modify: `src/hydra_suite/core/inference/runner.py` — add `run_batch_pass`, `_run_batch`, `load_frame`, `_load_headtail_for_indices`, `_load_cnn_for_indices`, `_load_pose_for_indices`, `_load_apriltag`
- Test: `tests/test_inference_runner_batch.py`

**Note:** The implementations of all seven additions are already written in Task 14's `runner.py` block above. Steps 1-5 of this task focus on the failing tests, driving the TDD cycle for the batch path independently.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_inference_runner_batch.py
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from hydra_suite.core.inference.result_types import OBBResult
from hydra_suite.core.inference.config import InferenceConfig


def _make_obb(n: int = 3, frame_idx: int = 0) -> OBBResult:
    rng = np.random.default_rng(0)
    return OBBResult(
        frame_idx=frame_idx,
        centers=rng.uniform(0, 640, (n, 2)).astype(np.float32),
        xywhr=rng.uniform(10, 100, (n, 5)).astype(np.float32),
        corners=rng.uniform(0, 640, (n, 4, 2)).astype(np.float32),
        conf=np.full(n, 0.9, dtype=np.float32),
    )


def test_run_batch_pass_raises_without_cache_dir():
    from hydra_suite.core.inference.runner import InferenceRunner
    cfg = InferenceConfig(compute_runtime="cpu")
    with patch("hydra_suite.core.inference.runner._load_all_models"):
        runner = InferenceRunner(cfg, cache_dir=None)
    with pytest.raises(RuntimeError, match="cache_dir"):
        runner.run_batch_pass(Path("video.mp4"))


def test_run_batch_pass_raises_on_unreadable_video(tmp_path):
    from hydra_suite.core.inference.runner import InferenceRunner
    cfg = InferenceConfig(compute_runtime="cpu")
    with patch("hydra_suite.core.inference.runner._load_all_models"):
        with patch("hydra_suite.core.inference.runner._open_caches") as mock_open:
            mock_caches = MagicMock()
            mock_caches.all_handles.return_value = []
            mock_open.return_value = mock_caches
            runner = InferenceRunner(cfg, cache_dir=tmp_path)
    with pytest.raises(IOError, match="Cannot open"):
        runner.run_batch_pass(tmp_path / "nonexistent.mp4")


def test_run_batch_pass_calls_progress_callback(tmp_path):
    import cv2
    from hydra_suite.core.inference.runner import InferenceRunner
    cfg = InferenceConfig(compute_runtime="cpu")

    mock_cap = MagicMock()
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.side_effect = [
        (True, fake_frame), (True, fake_frame), (True, fake_frame),
        (True, fake_frame), (True, fake_frame), (False, None),
    ]
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 5.0

    progress_calls: list[tuple] = []

    with patch("hydra_suite.core.inference.runner._load_all_models"):
        with patch("hydra_suite.core.inference.runner._open_caches") as mock_open:
            mock_caches = MagicMock()
            mock_caches.all_handles.return_value = []
            mock_open.return_value = mock_caches
            runner = InferenceRunner(cfg, cache_dir=tmp_path)
            runner._run_batch = MagicMock()
            with patch("cv2.VideoCapture", return_value=mock_cap):
                runner.run_batch_pass(
                    tmp_path / "video.mp4",
                    progress_cb=lambda done, total: progress_calls.append((done, total)),
                )
    assert len(progress_calls) > 0
    assert progress_calls[-1][1] == 5


def test_load_frame_raises_without_cache_dir():
    from hydra_suite.core.inference.runner import InferenceRunner
    cfg = InferenceConfig(compute_runtime="cpu")
    with patch("hydra_suite.core.inference.runner._load_all_models"):
        runner = InferenceRunner(cfg, cache_dir=None)
    with pytest.raises(RuntimeError, match="cache_dir"):
        runner.load_frame(0)


def test_load_frame_raises_on_missing_frame(tmp_path):
    from hydra_suite.core.inference.runner import InferenceRunner
    cfg = InferenceConfig(compute_runtime="cpu")
    with patch("hydra_suite.core.inference.runner._load_all_models"):
        with patch("hydra_suite.core.inference.runner._open_caches") as mock_open:
            mock_caches = MagicMock()
            mock_caches.detection.read_frame.return_value = None
            mock_caches.all_handles.return_value = [mock_caches.detection]
            mock_open.return_value = mock_caches
            runner = InferenceRunner(cfg, cache_dir=tmp_path)
    with pytest.raises(KeyError, match="0"):
        runner.load_frame(0)


def test_load_headtail_aligns_by_det_indices():
    from hydra_suite.core.inference.runner import _load_headtail_for_indices

    cached_det_indices = np.array([0, 1, 2, 3], dtype=np.int32)
    heading_hints = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    heading_confs = np.array([0.8, 0.9, 0.7, 0.95], dtype=np.float32)
    directed_mask = np.array([True, False, True, False])

    mock_cache = MagicMock()
    mock_cache.read_frame.return_value = (cached_det_indices, heading_hints, heading_confs, directed_mask)

    filtered_obb = _make_obb(2, frame_idx=7)
    det_indices = np.array([1, 3], dtype=np.int32)

    result = _load_headtail_for_indices(mock_cache, 7, det_indices, filtered_obb)
    assert result is not None
    np.testing.assert_allclose(result.heading_hints, [2.0, 4.0])
    np.testing.assert_allclose(result.heading_confidences, [0.9, 0.95])
    np.testing.assert_array_equal(result.directed_mask, [False, False])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_inference_runner_batch.py -v`
Expected: FAIL — `run_batch_pass`, `load_frame`, `_load_headtail_for_indices` not yet defined.

- [ ] **Step 3: Verify all implementations from Task 14 are in place**

All seven additions (`run_batch_pass`, `_run_batch`, `load_frame`, `_load_headtail_for_indices`, `_load_cnn_for_indices`, `_load_pose_for_indices`, `_load_apriltag`) are already in `runner.py` from Task 14, Step 5. No new code to write.

Run: `python -m pytest tests/test_inference_runner_batch.py -v`
Expected: all 6 tests pass.

- [ ] **Step 4: Run all Phase 4 tests together**

Run: `python -m pytest tests/test_inference_runner_rt.py tests/test_inference_runner_batch.py -v`
Expected: all 12 tests pass.

- [ ] **Step 5: Run full inference test suite**

Run: `python -m pytest tests/test_inference_config.py tests/test_inference_runtime_context.py tests/test_inference_result_types.py tests/test_inference_obb_stage.py tests/test_inference_filtering.py tests/test_inference_cache_keys.py tests/test_inference_cache_store.py tests/test_inference_runner_rt.py tests/test_inference_runner_batch.py -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_inference_runner_batch.py
git commit -m "test(inference): add batch pass and load_frame tests for InferenceRunner"
```

---

*Phase 4 complete. `InferenceRunner` handles model lifecycle, real-time inference, batch-pass caching, and threshold-decoupled frame loading via `det_indices` alignment. Proceed to Phase 5 -- Identity Evidence and worker.py integration.*

---

## Phase 5: Identity Evidence and worker.py Integration

**Goal:** Build `IdentityEvidenceBuilder` — the single place where temperature calibration, scoring-mode aggregation, and AprilTag→CNN priority resolution live — and wire `InferenceRunner` into `worker.py` behind a feature flag.

### Task 16: IdentityEvidenceBuilder

**Files:**
- Create: `src/hydra_suite/core/tracking/identity/evidence.py`
- Modify: `src/hydra_suite/core/tracking/identity/__init__.py`
- Test: `tests/test_identity_evidence_builder.py`

**Note on existing files:** `core/identity/evidence.py` already exists and is kept as-is (see spec Code Provenance). The new file is `core/tracking/identity/evidence.py` — a different package. Do not modify `core/identity/evidence.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_identity_evidence_builder.py
import numpy as np
import pytest
from unittest.mock import MagicMock

from hydra_suite.core.inference.result_types import (
    OBBResult, CNNResult, AprilTagResult, FrameResult,
)
from hydra_suite.core.inference.config import CNNConfig


def _make_frame_result(
    n_dets: int = 3,
    cnn_probs: np.ndarray | None = None,
    factor_names: list | None = None,
    class_names: list | None = None,
    apriltag_det_indices: list | None = None,
    apriltag_tag_ids: list | None = None,
) -> FrameResult:
    rng = np.random.default_rng(0)
    obb = OBBResult(
        frame_idx=0,
        centers=rng.uniform(0, 640, (n_dets, 2)).astype(np.float32),
        xywhr=rng.uniform(10, 100, (n_dets, 5)).astype(np.float32),
        corners=rng.uniform(0, 640, (n_dets, 4, 2)).astype(np.float32),
        conf=np.full(n_dets, 0.9, dtype=np.float32),
    )
    _factor_names = factor_names or ["identity"]
    _class_names = class_names or [["ant_A", "ant_B", "ant_C"]]
    n_classes = len(_class_names[0])
    n_factors = len(_factor_names)
    _probs = cnn_probs if cnn_probs is not None else rng.dirichlet(
        np.ones(n_classes), size=(n_dets, n_factors)
    ).astype(np.float32)[:, :, np.newaxis].repeat(n_classes, axis=2)

    cnn = CNNResult(
        frame_idx=0,
        probabilities=_probs,
        factor_names=_factor_names,
        class_names=_class_names,
    )

    apriltag = None
    if apriltag_det_indices is not None:
        n_tags = len(apriltag_det_indices)
        apriltag = AprilTagResult(
            tag_ids=apriltag_tag_ids or list(range(n_tags)),
            det_indices=apriltag_det_indices,
            centers=rng.uniform(0, 640, (n_tags, 2)).astype(np.float32),
            corners=rng.uniform(0, 640, (n_tags, 4, 2)).astype(np.float32),
        )

    return FrameResult(
        frame_idx=0,
        obb=obb,
        headtail=None,
        cnn=[cnn],
        pose=None,
        apriltag=apriltag,
    )


def _make_builder(label: str = "identity", temperature: float = 1.0):
    from hydra_suite.core.tracking.identity.evidence import IdentityEvidenceBuilder
    cfg = CNNConfig(label=label, model_path="/tmp/model.pt", calibration_temperature=temperature)
    catalog = MagicMock()
    catalog.get_label.return_value = None
    return IdentityEvidenceBuilder(cfg, catalog, phase_index=0)


def test_evidence_builder_produces_one_detection_per_obb():
    builder = _make_builder()
    frame = _make_frame_result(n_dets=3)
    evidence = builder.build(frame)
    assert len(evidence.detections) == 3


def test_evidence_builder_calibration_temperature_one_is_identity():
    # t=1 should not change winner
    rng = np.random.default_rng(42)
    probs = rng.dirichlet([5, 1, 1], size=(2, 1)).astype(np.float32)
    probs = probs[:, :, np.newaxis].repeat(3, axis=2)
    frame = _make_frame_result(n_dets=2, cnn_probs=probs)
    builder = _make_builder(temperature=1.0)
    evidence = builder.build(frame)
    for det in evidence.detections:
        assert det.cnn_factors is not None
        f = det.cnn_factors[0]
        assert f.winning_class == "ant_A"  # highest prior always wins


def test_evidence_builder_high_temperature_flattens_distribution():
    # t >> 1 makes distribution uniform; very high temperature should reduce confidence
    probs = np.array([[[0.99, 0.005, 0.005]]], dtype=np.float32).repeat(1, axis=0)
    frame = _make_frame_result(n_dets=1, cnn_probs=probs)
    hot_builder = _make_builder(temperature=100.0)
    cold_builder = _make_builder(temperature=0.1)
    hot_ev = hot_builder.build(frame)
    cold_ev = cold_builder.build(frame)
    assert hot_ev.detections[0].cnn_factors[0].confidence < cold_ev.detections[0].cnn_factors[0].confidence


def test_apriltag_overrides_cnn_when_both_present():
    from hydra_suite.core.tracking.identity.evidence import IdentityEvidenceBuilder
    cfg = CNNConfig(label="identity", model_path="/tmp/model.pt")
    catalog = MagicMock()
    catalog.get_label.return_value = "ant_A"  # tag maps to ant_A
    builder = IdentityEvidenceBuilder(cfg, catalog, phase_index=0)

    frame = _make_frame_result(n_dets=2, apriltag_det_indices=[0], apriltag_tag_ids=[7])
    evidence = builder.build(frame)

    det0 = evidence.detections[0]
    assert det0.is_authoritative is True
    assert det0.apriltag_label == "ant_A"
    assert det0.apriltag_tag_id == 7
    assert det0.resolved_label == "ant_A"

    det1 = evidence.detections[1]
    assert det1.is_authoritative is False
    assert det1.apriltag_label is None


def test_no_cnn_result_gives_none_factors():
    from hydra_suite.core.tracking.identity.evidence import IdentityEvidenceBuilder
    cfg = CNNConfig(label="identity", model_path="/tmp/model.pt")
    catalog = MagicMock()
    catalog.get_label.return_value = None
    builder = IdentityEvidenceBuilder(cfg, catalog, phase_index=5)  # index out of range

    frame = _make_frame_result(n_dets=2)
    evidence = builder.build(frame)
    for det in evidence.detections:
        assert det.cnn_factors is None
        assert det.resolved_label is None


def test_evidence_phase_label_matches_config():
    builder = _make_builder(label="my_phase")
    frame = _make_frame_result(n_dets=1)
    evidence = builder.build(frame)
    assert evidence.phase_label == "my_phase"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_identity_evidence_builder.py -v`
Expected: FAIL — `IdentityEvidenceBuilder` not yet defined.

- [ ] **Step 3: Create `core/tracking/identity/evidence.py`**

```python
# src/hydra_suite/core/tracking/identity/evidence.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hydra_suite.core.inference.config import CNNConfig
from hydra_suite.core.inference.result_types import FrameResult, AprilTagResult


@dataclass
class CNNFactorEvidence:
    factor_name: str
    class_names: list[str]
    calibrated_probabilities: np.ndarray   # (num_classes,) post-temperature softmax
    winning_class: str
    confidence: float


@dataclass
class DetectionIdentityEvidence:
    det_index: int
    cnn_factors: list[CNNFactorEvidence] | None  # None if CNN phase index is out of range
    apriltag_label: str | None
    apriltag_tag_id: int | None
    resolved_label: str | None           # apriltag overrides cnn when both present
    resolved_confidence: float
    is_authoritative: bool               # True = label came from AprilTag


@dataclass
class FrameIdentityEvidence:
    frame_idx: int
    phase_label: str
    detections: list[DetectionIdentityEvidence]


class IdentityEvidenceBuilder:
    """Converts raw FrameResult into FrameIdentityEvidence for one CNN phase.

    Applies temperature calibration, scoring_mode aggregation, and AprilTag
    priority resolution. One instance per CNN phase; worker.py creates one
    per enabled phase in config.cnn and calls build() once per frame.
    """

    def __init__(self, config: CNNConfig, catalog: Any, phase_index: int) -> None:
        self.config = config
        self._catalog = catalog
        self._phase_index = phase_index

    def build(self, frame_result: FrameResult) -> FrameIdentityEvidence:
        cnn_result = (
            frame_result.cnn[self._phase_index]
            if self._phase_index < len(frame_result.cnn) else None
        )
        n_dets = len(frame_result.obb.centers) if frame_result.obb is not None else 0
        detections = [
            self._build_detection(det_idx, cnn_result, frame_result.apriltag)
            for det_idx in range(n_dets)
        ]
        return FrameIdentityEvidence(
            frame_idx=frame_result.frame_idx,
            phase_label=self.config.label,
            detections=detections,
        )

    def _build_detection(
        self,
        det_idx: int,
        cnn_result,
        apriltag_result: AprilTagResult | None,
    ) -> DetectionIdentityEvidence:
        cnn_factors = self._build_cnn_factors(det_idx, cnn_result)
        apriltag_label, apriltag_tag_id = self._lookup_apriltag(det_idx, apriltag_result)

        if apriltag_label is not None:
            return DetectionIdentityEvidence(
                det_index=det_idx,
                cnn_factors=cnn_factors,
                apriltag_label=apriltag_label,
                apriltag_tag_id=apriltag_tag_id,
                resolved_label=apriltag_label,
                resolved_confidence=1.0,
                is_authoritative=True,
            )

        resolved_label, resolved_confidence = self._resolve_from_cnn(cnn_factors)
        return DetectionIdentityEvidence(
            det_index=det_idx,
            cnn_factors=cnn_factors,
            apriltag_label=None,
            apriltag_tag_id=None,
            resolved_label=resolved_label,
            resolved_confidence=resolved_confidence,
            is_authoritative=False,
        )

    def _build_cnn_factors(self, det_idx: int, cnn_result) -> list[CNNFactorEvidence] | None:
        if cnn_result is None:
            return None
        probs_3d = cnn_result.probabilities  # (M, F, C_max) float32
        if det_idx >= len(probs_3d):
            return None
        factor_probs = probs_3d[det_idx]    # (F, C_max)
        factors: list[CNNFactorEvidence] = []
        for f_idx, factor_name in enumerate(cnn_result.factor_names):
            class_names = cnn_result.class_names[f_idx]
            n_classes = len(class_names)
            raw = factor_probs[f_idx, :n_classes]
            calibrated = self._calibrate(raw)
            winning_idx = int(np.argmax(calibrated))
            factors.append(CNNFactorEvidence(
                factor_name=factor_name,
                class_names=class_names,
                calibrated_probabilities=calibrated,
                winning_class=class_names[winning_idx],
                confidence=float(calibrated[winning_idx]),
            ))
        return factors

    def _calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        t = self.config.calibration_temperature
        if abs(t - 1.0) < 1e-6:
            s = raw_probs.sum()
            return raw_probs / s if s > 0 else raw_probs
        log_probs = np.log(raw_probs + 1e-10)
        scaled = log_probs / t
        scaled -= scaled.max()
        out = np.exp(scaled)
        return (out / out.sum()).astype(np.float32)

    def _lookup_apriltag(
        self,
        det_idx: int,
        apriltag_result: AprilTagResult | None,
    ) -> tuple[str | None, int | None]:
        if apriltag_result is None:
            return None, None
        for i, di in enumerate(apriltag_result.det_indices):
            if di == det_idx:
                tag_id = apriltag_result.tag_ids[i]
                label = self._catalog.get_label(tag_id)
                return label, tag_id
        return None, None

    def _resolve_from_cnn(
        self, factors: list[CNNFactorEvidence] | None
    ) -> tuple[str | None, float]:
        if not factors:
            return None, 0.0
        if self.config.scoring_mode == "atomic" or len(factors) == 1:
            best = max(factors, key=lambda f: f.confidence)
            return best.winning_class, best.confidence
        # per_head_average: pick the class that wins in the most factors;
        # break ties by summed calibrated probability for that class.
        from collections import Counter
        votes: Counter = Counter(f.winning_class for f in factors)
        top_class, _ = votes.most_common(1)[0]
        avg_conf = float(np.mean([
            f.calibrated_probabilities[f.class_names.index(top_class)]
            for f in factors
            if top_class in f.class_names
        ]))
        return top_class, avg_conf
```

Fix the missing `Any` import at the top of the file:

```python
from typing import Any
```

- [ ] **Step 4: Update `core/tracking/identity/__init__.py`**

```python
from .evidence import (
    CNNFactorEvidence,
    DetectionIdentityEvidence,
    FrameIdentityEvidence,
    IdentityEvidenceBuilder,
)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_identity_evidence_builder.py -v`
Expected: all 6 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/tracking/identity/evidence.py \
        src/hydra_suite/core/tracking/identity/__init__.py \
        tests/test_identity_evidence_builder.py
git commit -m "feat(inference): add IdentityEvidenceBuilder with temperature calibration and AprilTag priority"
```

---

### Task 17: worker.py integration — new inference pipeline path

**Files:**
- Modify: `src/hydra_suite/core/tracking/worker.py`
- Test: `tests/test_worker_inference_integration.py`

**Strategy:** Add a `USE_NEW_INFERENCE_PIPELINE = True` feature flag at the top of worker.py. Add a `_run_with_new_pipeline()` method and a `_run_realtime_with_new_pipeline()` method. The existing `run()` method dispatches to either path based on the flag. This allows output verification before deleting old code.

- [ ] **Step 1: Identify the dispatch point in `worker.py`**

Open `src/hydra_suite/core/tracking/worker.py`. Find the top-level `run()` method. Identify:
- Where `self.params` (or equivalent config dict) is read
- Where the main tracking loop begins
- Where the backward pass begins

Note the line numbers of these three anchor points before making changes.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_worker_inference_integration.py
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path


def _make_frame_result():
    from hydra_suite.core.inference.result_types import OBBResult, FrameResult
    obb = OBBResult(
        frame_idx=0,
        centers=np.zeros((2, 2), dtype=np.float32),
        xywhr=np.zeros((2, 5), dtype=np.float32),
        corners=np.zeros((2, 4, 2), dtype=np.float32),
        conf=np.array([0.9, 0.8], dtype=np.float32),
    )
    return FrameResult(frame_idx=0, obb=obb, headtail=None, cnn=[], pose=None, apriltag=None)


def test_worker_uses_new_pipeline_flag():
    """Verify USE_NEW_INFERENCE_PIPELINE constant exists and is True."""
    from hydra_suite.core.tracking import worker
    assert hasattr(worker, "USE_NEW_INFERENCE_PIPELINE")
    assert worker.USE_NEW_INFERENCE_PIPELINE is True


def test_run_with_new_pipeline_calls_batch_pass_when_caches_invalid(tmp_path):
    """When caches are invalid, run_batch_pass is called before the tracking loop."""
    from hydra_suite.core.tracking.worker import TrackingWorker

    mock_runner = MagicMock()
    mock_runner.caches_all_valid.return_value = False
    mock_runner.load_frame.return_value = _make_frame_result()

    with patch("hydra_suite.core.tracking.worker.InferenceRunner", return_value=mock_runner):
        with patch("hydra_suite.core.tracking.worker.InferenceConfig") as mock_cfg_cls:
            mock_cfg_cls.from_json.return_value = MagicMock(realtime=False, cnn=[])
            worker_obj = TrackingWorker.__new__(TrackingWorker)
            worker_obj._identity_builders = []
            worker_obj._run_with_new_pipeline(
                video_path=tmp_path / "video.mp4",
                config_path=str(tmp_path / "cfg.json"),
                cache_dir=tmp_path,
                total_frames=2,
            )

    mock_runner.run_batch_pass.assert_called_once()


def test_run_with_new_pipeline_skips_batch_pass_when_caches_valid(tmp_path):
    """When caches are valid, run_batch_pass is NOT called."""
    from hydra_suite.core.tracking.worker import TrackingWorker

    mock_runner = MagicMock()
    mock_runner.caches_all_valid.return_value = True
    mock_runner.load_frame.return_value = _make_frame_result()

    with patch("hydra_suite.core.tracking.worker.InferenceRunner", return_value=mock_runner):
        with patch("hydra_suite.core.tracking.worker.InferenceConfig") as mock_cfg_cls:
            mock_cfg_cls.from_json.return_value = MagicMock(realtime=False, cnn=[])
            worker_obj = TrackingWorker.__new__(TrackingWorker)
            worker_obj._identity_builders = []
            worker_obj._run_with_new_pipeline(
                video_path=tmp_path / "video.mp4",
                config_path=str(tmp_path / "cfg.json"),
                cache_dir=tmp_path,
                total_frames=2,
            )

    mock_runner.run_batch_pass.assert_not_called()


def test_run_realtime_calls_run_realtime_per_frame(tmp_path):
    """RT path calls run_realtime() once per frame."""
    from hydra_suite.core.tracking.worker import TrackingWorker
    import cv2

    frames = [np.zeros((480, 640, 3), dtype=np.uint8)] * 3
    mock_runner = MagicMock()
    mock_runner.run_realtime.return_value = _make_frame_result()

    with patch("hydra_suite.core.tracking.worker.InferenceRunner", return_value=mock_runner):
        with patch("hydra_suite.core.tracking.worker.InferenceConfig") as mock_cfg_cls:
            mock_cfg_cls.from_json.return_value = MagicMock(realtime=True, cnn=[])
            worker_obj = TrackingWorker.__new__(TrackingWorker)
            worker_obj._identity_builders = []
            worker_obj._run_realtime_with_new_pipeline(frames, MagicMock())

    assert mock_runner.run_realtime.call_count == 3
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_worker_inference_integration.py -v`
Expected: FAIL — `USE_NEW_INFERENCE_PIPELINE`, `_run_with_new_pipeline`, `_run_realtime_with_new_pipeline` not yet defined.

- [ ] **Step 4: Add feature flag and imports to `worker.py`**

At the very top of `worker.py`, after the existing imports block, add:

```python
# Feature flag: route run() through the new InferenceRunner-based pipeline.
# Set to False to revert to legacy detection_phase / precompute path for debugging.
USE_NEW_INFERENCE_PIPELINE = True

from hydra_suite.core.inference.runner import InferenceRunner
from hydra_suite.core.inference.config import InferenceConfig
```

- [ ] **Step 5: Add `_resolve_cache_dir`, `_run_with_new_pipeline`, and `_run_realtime_with_new_pipeline` to `TrackingWorker`**

Add these three methods to the `TrackingWorker` class. Locate the end of the class (before any standalone functions) and add:

```python
    def _resolve_cache_dir(self) -> Path:
        """Return the per-video cache directory for InferenceRunner caches."""
        video_path = Path(self.video_path)
        return video_path.parent / f".inference_cache_{video_path.stem}"

    def _run_with_new_pipeline(
        self,
        video_path: Path,
        config_path: str,
        cache_dir: Path,
        total_frames: int,
    ) -> None:
        """Non-RT tracking pass using InferenceRunner.

        Runs the inference batch pass if any cache is invalid, then drives the
        tracking loop by loading pre-computed FrameResult objects from cache.
        Kalman, assignment, backward pass, and consensus resolution are unchanged.
        """
        config = InferenceConfig.from_json(config_path)
        runner = InferenceRunner(config, cache_dir=cache_dir)

        try:
            if not runner.caches_all_valid():
                runner.run_batch_pass(
                    video_path,
                    progress_cb=lambda done, total: self._emit_inference_progress(done, total),
                )

            for frame_idx in range(total_frames):
                frame_result = runner.load_frame(frame_idx)
                for builder in self._identity_builders:
                    evidence = builder.build(frame_result)
                    self._apply_identity_evidence(frame_idx, evidence)
                self._update_tracks(frame_idx, frame_result)
        finally:
            runner.close()

    def _run_realtime_with_new_pipeline(
        self,
        frames: list,
        runner: InferenceRunner,
    ) -> None:
        """RT tracking pass using InferenceRunner.run_realtime() per frame."""
        for frame_idx, frame in enumerate(frames):
            frame_result = runner.run_realtime(frame)
            for builder in self._identity_builders:
                evidence = builder.build(frame_result)
                self._apply_identity_evidence(frame_idx, evidence)
            self._update_tracks(frame_idx, frame_result)
```

- [ ] **Step 6: Wire the dispatch in `run()`**

Find the section in `run()` where inference and tracking begin (after parameter validation, before the detection loop). Add a conditional dispatch:

```python
        if USE_NEW_INFERENCE_PIPELINE:
            if self._is_realtime_mode():
                config = InferenceConfig.from_json(self._get_config_path())
                runner = InferenceRunner(config, cache_dir=None)
                try:
                    self._run_realtime_with_new_pipeline(self._frame_iterator(), runner)
                finally:
                    runner.close()
            else:
                self._run_with_new_pipeline(
                    video_path=Path(self.video_path),
                    config_path=self._get_config_path(),
                    cache_dir=self._resolve_cache_dir(),
                    total_frames=self._total_frames,
                )
            return
        # Legacy path continues below...
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_worker_inference_integration.py -v`
Expected: all 4 tests pass.

- [ ] **Step 8: Run full test suite to check for regressions**

Run: `python -m pytest tests/ -v -m "not benchmark"`
Expected: all tests pass. Any failures in existing tests indicate regressions from the worker.py edit — fix before committing.

- [ ] **Step 9: Commit**

```bash
git add src/hydra_suite/core/tracking/worker.py \
        tests/test_worker_inference_integration.py
git commit -m "feat(inference): wire InferenceRunner into worker.py behind USE_NEW_INFERENCE_PIPELINE flag"
```

---

### Task 18: Verification, flag removal, and legacy file deletion

**Files:**
- Modify: `src/hydra_suite/core/tracking/worker.py` — remove feature flag, always use new path
- Delete: 16 legacy files listed in spec (see below)
- Test: verification script comparing old vs new cache outputs

**Prerequisite:** Run the full pipeline on at least one real video with `USE_NEW_INFERENCE_PIPELINE = True` and verify that trajectory outputs match the legacy path frame-by-frame before proceeding to deletion.

- [ ] **Step 1: Run output comparison on a test video**

```bash
# Set USE_NEW_INFERENCE_PIPELINE = False, run on a test video, save trajectories
python -c "
from hydra_suite.core.tracking.worker import TrackingWorker
# ... run worker on test_video.mp4, save CSV to /tmp/legacy_output.csv
"

# Set USE_NEW_INFERENCE_PIPELINE = True, delete caches, run again
python -c "
from hydra_suite.core.tracking.worker import TrackingWorker
# ... run worker on same video, save CSV to /tmp/new_output.csv
"

# Diff
python -c "
import pandas as pd
old = pd.read_csv('/tmp/legacy_output.csv')
new = pd.read_csv('/tmp/new_output.csv')
diff = (old - new).abs().max()
print(diff)
assert (diff < 1e-4).all(), 'Output mismatch -- do not delete legacy files'
print('Outputs match. Safe to delete legacy files.')
"
```

Expected: all columns match within floating-point tolerance.

- [ ] **Step 2: Remove the feature flag from `worker.py`**

Find in `worker.py`:

```python
# Feature flag: route run() through the new InferenceRunner-based pipeline.
# Set to False to revert to legacy detection_phase / precompute path for debugging.
USE_NEW_INFERENCE_PIPELINE = True
```

Remove the flag and the conditional dispatch block in `run()`. The `_run_with_new_pipeline` and `_run_realtime_with_new_pipeline` methods become the unconditional implementation.

Also remove the test `test_worker_uses_new_pipeline_flag` from `tests/test_worker_inference_integration.py` since the flag no longer exists.

- [ ] **Step 3: Delete legacy inference files**

```bash
git rm src/hydra_suite/core/tracking/detection_phase.py
git rm src/hydra_suite/core/tracking/precompute.py
git rm src/hydra_suite/core/tracking/pose_pipeline.py
git rm src/hydra_suite/core/tracking/live_features.py
git rm src/hydra_suite/core/tracking/cnn_features.py
git rm src/hydra_suite/core/tracking/tag_features.py
git rm src/hydra_suite/core/tracking/evidence_emitter.py
git rm src/hydra_suite/core/detectors/yolo_detector.py
git rm src/hydra_suite/core/detectors/factory.py
git rm src/hydra_suite/core/detectors/detection_filter.py
git rm src/hydra_suite/core/identity/classification/cnn.py
git rm src/hydra_suite/core/identity/classification/headtail.py
git rm src/hydra_suite/core/identity/pose/api.py
git rm src/hydra_suite/core/identity/pose/backends/yolo.py
git rm src/hydra_suite/core/identity/pose/backends/sleap.py
git rm src/hydra_suite/core/identity/pose/backends/sleap_utils.py
git rm src/hydra_suite/data/detection_cache.py
git rm src/hydra_suite/data/tag_observation_cache.py
git rm src/hydra_suite/core/identity/properties/cache.py
git rm src/hydra_suite/core/identity/properties/detected_cache.py
```

- [ ] **Step 4: Fix any broken imports caused by deletions**

Run: `python -m pytest tests/ -v -m "not benchmark" 2>&1 | grep "ImportError\|ModuleNotFoundError"`
Expected: no import errors. If any appear, trace the import chain and remove or redirect the broken import.

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v -m "not benchmark"`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add -u
git commit -m "feat(inference): remove feature flag, delete legacy inference files, complete pipeline migration"
```

---

*Phase 5 complete. `IdentityEvidenceBuilder` handles calibration, scoring, and AprilTag priority. `worker.py` delegates all inference to `InferenceRunner`. Legacy inference files deleted. The `core/inference/` redesign is fully integrated.*

---

## Type Consistency Corrections

**Read this before implementing Phase 4 or Phase 5.**

The Phase 4-5 code was written against a simplified mental model of the result types. Phase 1 defines the authoritative types. Reconcile these differences before running Phase 4's Step 2.

### OBBResult field names

Phase 1 defines:

```python
@dataclass
class OBBResult:
    centroids: np.ndarray      # (D, 2) cx, cy  -- NOT 'centers'
    angles: np.ndarray         # (D,) radians
    sizes: np.ndarray          # (D,) area px²   -- replaces xywhr[:,2]*xywhr[:,3]
    shapes: np.ndarray         # (D, 2) ellipse_area, aspect_ratio
    confidences: np.ndarray    # (D,)             -- NOT 'conf'
    corners: np.ndarray        # (D, 4, 2)

    @property
    def num_detections(self) -> int:
        return int(len(self.confidences))
```

In every test and implementation in Phase 4-5, apply these substitutions:

| Phase 4-5 wrote | Correct Phase 1 name |
|---|---|
| `raw.conf` | `raw.confidences` |
| `raw.centers` | `raw.centroids` |
| `raw.xywhr[:, 2] * raw.xywhr[:, 3]` | `raw.sizes` |
| `len(obb.centers)` | `obb.num_detections` |
| `OBBResult(frame_idx=..., centers=..., xywhr=..., corners=..., conf=...)` | `OBBResult(centroids=..., angles=..., sizes=..., shapes=..., confidences=..., corners=...)` |

**Corrected `_make_obb` test helper** (replaces all `_make_obb` definitions in Phase 4-5 tests):

```python
def _make_obb(n: int = 5, conf_values: list | None = None) -> OBBResult:
    rng = np.random.default_rng(42)
    confidences = (
        np.array(conf_values, dtype=np.float32) if conf_values is not None
        else rng.uniform(0.2, 1.0, n).astype(np.float32)
    )
    n_actual = len(confidences)
    w = rng.uniform(10, 50, n_actual).astype(np.float32)
    h = rng.uniform(20, 80, n_actual).astype(np.float32)
    return OBBResult(
        centroids=rng.uniform(0, 640, (n_actual, 2)).astype(np.float32),
        angles=rng.uniform(0, np.pi, n_actual).astype(np.float32),
        sizes=(w * h).astype(np.float32),
        shapes=np.stack([w * h, h / w], axis=1).astype(np.float32),
        confidences=confidences,
        corners=rng.uniform(0, 640, (n_actual, 4, 2)).astype(np.float32),
    )
```

**Corrected `filter_with_indices`** Step 3 body (replaces Step 3 in Task 14):

```python
def filter_with_indices(
    raw: OBBResult,
    config: OBBConfig,
    roi_mask: np.ndarray | None = None,
) -> tuple[OBBResult, np.ndarray]:
    """Run the same gates as filter_detections and return (filtered_result, pre-filter indices).

    Returned indices index into *raw*. They are stored alongside downstream
    results in the cache so that a threshold edit never invalidates those caches.
    """
    keep = raw.confidences >= config.confidence_threshold
    if config.min_object_size > 0:
        keep = keep & (raw.sizes >= config.min_object_size)
    if roi_mask is not None:
        h, w = roi_mask.shape[:2]
        cx = np.clip(raw.centroids[:, 0].astype(np.int32), 0, w - 1)
        cy = np.clip(raw.centroids[:, 1].astype(np.int32), 0, h - 1)
        keep = keep & roi_mask[cy, cx].astype(bool)
    indices = np.where(keep)[0]
    subset = _select(raw, indices)
    if config.nms_iou_threshold < 1.0 and len(indices) > 1:
        keep_nms = _apply_nms(subset, config.nms_iou_threshold)
        indices = indices[keep_nms]
        subset = _select(raw, indices)
    return subset, indices.astype(np.int32)
```

**Corrected `filter_with_indices` tests** (replaces the test function bodies in Task 14 Step 1):

```python
def test_filter_with_indices_returns_correct_indices():
    from hydra_suite.core.inference.filtering import filter_with_indices
    raw = _make_obb(conf_values=[0.9, 0.1, 0.8, 0.2, 0.7])
    cfg = OBBConfig(confidence_threshold=0.5, min_object_size=0, nms_iou_threshold=1.0)
    filtered, indices = filter_with_indices(raw, cfg)
    assert set(indices.tolist()) == {0, 2, 4}
    assert filtered.num_detections == 3


def test_filter_with_indices_centers_match_original():
    from hydra_suite.core.inference.filtering import filter_with_indices
    raw = _make_obb(conf_values=[0.9, 0.1, 0.8, 0.2, 0.7])
    cfg = OBBConfig(confidence_threshold=0.5, min_object_size=0, nms_iou_threshold=1.0)
    filtered, indices = filter_with_indices(raw, cfg)
    for i, orig_idx in enumerate(indices):
        np.testing.assert_allclose(filtered.centroids[i], raw.centroids[orig_idx])
        np.testing.assert_allclose(filtered.confidences[i], raw.confidences[orig_idx])


def test_filter_with_indices_empty_result():
    from hydra_suite.core.inference.filtering import filter_with_indices
    raw = _make_obb(conf_values=[0.1, 0.2, 0.1])
    cfg = OBBConfig(confidence_threshold=0.5, min_object_size=0, nms_iou_threshold=1.0)
    filtered, indices = filter_with_indices(raw, cfg)
    assert len(indices) == 0
    assert filtered.num_detections == 0
    assert indices.dtype == np.int32
```

### FrameResult required fields

Phase 1 defines `FrameResult` with two additional required fields not included in Phase 4's `_build_frame_result`:

```python
@dataclass
class FrameResult:
    frame_idx: int
    obb: OBBResult
    filtered_indices: list[int]        # det_indices that survived filtering
    headtail: HeadTailResult | None
    cnn: list[CNNResult]
    pose: PoseResult | None
    apriltag: AprilTagResult | None
    resolved_headings: np.ndarray      # (D,) final merged heading per detection
```

**Corrected `_build_frame_result`** (replaces the function in Task 14 Step 5 of `runner.py`):

```python
def _resolve_headings(
    obb: OBBResult,
    ht: HeadTailResult | None,
    pose: PoseResult | None,
) -> np.ndarray:
    """Merge headings: pose > headtail > OBB axis (fallback to 0.0)."""
    n = obb.num_detections
    headings = np.full(n, 0.0, dtype=np.float32)
    if ht is not None:
        valid = ht.directed_mask.astype(bool)
        headings[valid] = ht.heading_hints[valid]
    if pose is not None and hasattr(pose, "heading_overrides"):
        valid_pose = pose.valid_mask.astype(bool)
        headings[valid_pose] = pose.heading_overrides[valid_pose]
    return headings


def _build_frame_result(
    frame_idx: int,
    filtered_obb: OBBResult,
    det_indices: np.ndarray,
    ht_result: HeadTailResult | None,
    cnn_results: list,
    pose_result: PoseResult | None,
    apriltag_result: AprilTagResult | None,
) -> FrameResult:
    return FrameResult(
        frame_idx=frame_idx,
        obb=filtered_obb,
        filtered_indices=list(det_indices),
        headtail=ht_result,
        cnn=cnn_results,
        pose=pose_result,
        apriltag=apriltag_result,
        resolved_headings=_resolve_headings(filtered_obb, ht_result, pose_result),
    )
```

Update every call site of `_build_frame_result` in `runner.py` to pass `det_indices`:

```python
# In run_realtime (early return when no detections):
return _build_frame_result(0, filtered_obb, np.zeros(0, np.int32), None, [], None, None)

# Normal run_realtime return:
return _build_frame_result(0, filtered_obb, det_indices, ht_result, cnn_results, pose_result, at_result)

# In load_frame:
return _build_frame_result(frame_idx, filtered_obb, det_indices, ht_result, cnn_results, pose_result, at_result)
```

### CNNResult format

Phase 1 uses the rich prediction format with nested dataclasses, NOT a flat 3D numpy array:

```python
@dataclass
class CNNResult:
    label: str
    predictions: list[CNNDetectionPrediction]  # one per surviving detection
```

**Corrected `_load_cnn_for_indices`** (replaces the function in Task 14 Step 5 of `runner.py`):

```python
def _load_cnn_for_indices(
    caches: list[CNNCacheHandle],
    cnn_configs: list,           # list[CNNConfig] — needed for the label field
    frame_idx: int,
    det_indices: np.ndarray,
) -> list[CNNResult]:
    results: list[CNNResult] = []
    det_set = set(int(di) for di in det_indices)
    for cache, cfg in zip(caches, cnn_configs):
        preds: list[CNNDetectionPrediction] | None = cache.read_frame(frame_idx)
        if preds is None:
            results.append(None)
            continue
        aligned = [p for p in preds if p.det_index in det_set]
        results.append(CNNResult(label=cfg.label, predictions=aligned))
    return results
```

Update `load_frame` to pass CNN configs:

```python
    def load_frame(self, frame_idx: int) -> FrameResult:
        ...
        enabled_cnn_cfgs = [c for c in self.config.cnn if c.enabled]
        cnn_results = _load_cnn_for_indices(self._caches.cnn, enabled_cnn_cfgs, frame_idx, det_indices)
        ...
```

### HeadTailResult optional canonical_affines

Phase 1 includes `canonical_affines: np.ndarray` in `HeadTailResult`. The cache-loaded path cannot reconstruct this. Make it optional by defaulting to `None`:

```python
# When loading from cache in _load_headtail_for_indices, omit canonical_affines:
return HeadTailResult(
    heading_hints=out_hints,
    heading_confidences=out_confs,
    directed_mask=out_directed,
    canonical_affines=np.zeros((n, 2, 3), dtype=np.float32),  # zeros sentinel for cache path
)
```

### Phase 5 IdentityEvidenceBuilder corrections

**Corrected `_build_cnn_factors`** uses Phase 1's `CNNResult.predictions` format:

```python
    def _build_cnn_factors(self, det_idx: int, cnn_result: CNNResult | None) -> list[CNNFactorEvidence] | None:
        if cnn_result is None:
            return None
        # Find the prediction for this detection index
        pred = next((p for p in cnn_result.predictions if p.det_index == det_idx), None)
        if pred is None:
            return None
        factors: list[CNNFactorEvidence] = []
        for factor in pred.factors:
            calibrated = self._calibrate(factor.raw_probabilities)
            winning_idx = int(np.argmax(calibrated))
            factors.append(CNNFactorEvidence(
                factor_name=factor.factor_name,
                class_names=factor.class_names,
                calibrated_probabilities=calibrated,
                winning_class=factor.class_names[winning_idx],
                confidence=float(calibrated[winning_idx]),
            ))
        return factors
```

**Corrected Phase 5 test `_make_frame_result`** (replaces the helper in `test_identity_evidence_builder.py`):

```python
def _make_frame_result(
    n_dets: int = 3,
    raw_probs_per_det: list | None = None,   # list[np.ndarray] one (num_classes,) per det
    factor_name: str = "identity",
    class_names: list | None = None,
    apriltag_det_indices: list | None = None,
    apriltag_tag_ids: list | None = None,
) -> FrameResult:
    rng = np.random.default_rng(0)
    _class_names = class_names or ["ant_A", "ant_B", "ant_C"]
    n_classes = len(_class_names)
    w = rng.uniform(10, 50, n_dets).astype(np.float32)
    h = rng.uniform(20, 80, n_dets).astype(np.float32)
    obb = OBBResult(
        centroids=rng.uniform(0, 640, (n_dets, 2)).astype(np.float32),
        angles=rng.uniform(0, np.pi, n_dets).astype(np.float32),
        sizes=(w * h).astype(np.float32),
        shapes=np.stack([w * h, h / w], axis=1).astype(np.float32),
        confidences=np.full(n_dets, 0.9, dtype=np.float32),
        corners=rng.uniform(0, 640, (n_dets, 4, 2)).astype(np.float32),
    )
    probs_list = raw_probs_per_det or [
        rng.dirichlet(np.ones(n_classes)).astype(np.float32) for _ in range(n_dets)
    ]
    predictions = [
        CNNDetectionPrediction(
            det_index=i,
            factors=[CNNFactorPrediction(
                factor_name=factor_name,
                class_names=_class_names,
                raw_probabilities=probs_list[i],
            )],
        )
        for i in range(n_dets)
    ]
    cnn = CNNResult(label="identity", predictions=predictions)

    apriltag = None
    if apriltag_det_indices is not None:
        n_tags = len(apriltag_det_indices)
        apriltag = AprilTagResult(
            tag_ids=apriltag_tag_ids or list(range(n_tags)),
            det_indices=apriltag_det_indices,
            centers=rng.uniform(0, 640, (n_tags, 2)).astype(np.float32),
            corners=rng.uniform(0, 640, (n_tags, 4, 2)).astype(np.float32),
        )

    return FrameResult(
        frame_idx=0,
        obb=obb,
        filtered_indices=list(range(n_dets)),
        headtail=None,
        cnn=[cnn],
        pose=None,
        apriltag=apriltag,
        resolved_headings=np.zeros(n_dets, dtype=np.float32),
    )
```

Add these imports to `test_identity_evidence_builder.py`:

```python
from hydra_suite.core.inference.result_types import (
    OBBResult, CNNResult, CNNDetectionPrediction, CNNFactorPrediction,
    AprilTagResult, FrameResult,
)
```

---

*All type inconsistencies between Phase 1 definitions and Phase 4-5 implementations are now documented. Implement Phase 4 and Phase 5 using the corrected code in this section wherever it conflicts with the task steps above.*

---

## Bug and Performance Corrections

**Read this before implementing any task from Phase 1 onwards.**

Thirteen correctness bugs and performance issues were identified in review. Each correction below supersedes the corresponding code block in the phase tasks. Apply them in order of their task number.

---

### Correction 1 — Task 3: Add `tensor_on_cuda` to `RuntimeContext`

`cuda_mode` is True for all three CUDA-group runtimes (`cuda`, `onnx_cuda`, `tensorrt`). But only native PyTorch `cuda` leaves model outputs as live CUDA device tensors. ONNX Runtime (`onnx_cuda`) and TensorRT both produce CPU numpy arrays from the inference call regardless of which execution provider they use. `_RawOBBTensors` must only be used when the model returns actual CUDA tensors.

**Corrected `runtime.py` (replaces Task 3 Step 3)**:

```python
from __future__ import annotations

from dataclasses import dataclass, field

from .config import InferenceConfig, ComputeRuntime, CUDA_RUNTIMES


@dataclass(frozen=True)
class RuntimeContext:
    cuda_mode: bool
    device: str               # "cuda:0", "mps", or "cpu"
    use_nvdec: bool           # cuda_mode AND NVDEC available
    default_runtime: ComputeRuntime
    tensor_on_cuda: bool = False  # True ONLY for native PyTorch "cuda" runtime;
                                  # onnx_cuda and tensorrt return CPU numpy despite GPU use

    @staticmethod
    def from_config(config: InferenceConfig) -> "RuntimeContext":
        runtimes = config._collect_all_runtimes()
        cuda_mode = bool(runtimes & CUDA_RUNTIMES)
        # tensor_on_cuda: native "cuda" runtime leaves model outputs as CUDA tensors.
        # "onnx_cuda" and "tensorrt" use GPU compute but return CPU numpy — NOT CUDA tensors.
        tensor_on_cuda = "cuda" in runtimes and not (runtimes & {"onnx_cuda", "tensorrt"})
        if cuda_mode:
            device = _cuda_device_available()
            nvdec = _nvdec_available()
        else:
            device = _cpu_or_mps_device()
            nvdec = False
        default: ComputeRuntime = "cuda" if cuda_mode else "cpu"
        return RuntimeContext(
            cuda_mode=cuda_mode,
            device=device,
            use_nvdec=nvdec,
            default_runtime=default,
            tensor_on_cuda=tensor_on_cuda,
        )


def _cuda_device_available() -> str:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA runtime requested but no CUDA device is available. "
            "Check your CUDA installation or switch to a CPU-group runtime."
        )
    return "cuda:0"


def _cpu_or_mps_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _nvdec_available() -> bool:
    try:
        import torchvision
        return torchvision.get_video_backend() == "cuda"
    except Exception:
        return False
```

**Add one test** to `tests/test_inference_runtime.py`:

```python
def test_tensor_on_cuda_true_only_for_native_cuda():
    # tensor_on_cuda True: pure cuda runtime only
    with patch("hydra_suite.core.inference.runtime._cuda_device_available",
               return_value="cuda:0"):
        with patch("hydra_suite.core.inference.runtime._nvdec_available",
                   return_value=False):
            ctx = RuntimeContext.from_config(_cuda_config())
    assert ctx.tensor_on_cuda is True

    # tensor_on_cuda False: onnx_cuda uses GPU but returns CPU numpy
    onnx_cuda_cfg = InferenceConfig(
        obb=OBBConfig(mode="direct",
                      direct=OBBDirectConfig(model_path="/m.onnx", compute_runtime="onnx_cuda"))
    )
    with patch("hydra_suite.core.inference.runtime._cuda_device_available",
               return_value="cuda:0"):
        with patch("hydra_suite.core.inference.runtime._nvdec_available",
                   return_value=False):
            ctx2 = RuntimeContext.from_config(onnx_cuda_cfg)
    assert ctx2.cuda_mode is True       # CUDA group
    assert ctx2.tensor_on_cuda is False  # but outputs are CPU numpy

    # tensor_on_cuda False: tensorrt also returns CPU numpy
    trt_cfg = InferenceConfig(
        obb=OBBConfig(mode="direct",
                      direct=OBBDirectConfig(model_path="/m.engine", compute_runtime="tensorrt"))
    )
    with patch("hydra_suite.core.inference.runtime._cuda_device_available",
               return_value="cuda:0"):
        with patch("hydra_suite.core.inference.runtime._nvdec_available",
                   return_value=False):
            ctx3 = RuntimeContext.from_config(trt_cfg)
    assert ctx3.cuda_mode is True
    assert ctx3.tensor_on_cuda is False
```

Update every direct `RuntimeContext(...)` construction in tests (Tasks 5, 6, 7) to pass `tensor_on_cuda=True` for CUDA helpers:

```python
def _cuda_rt() -> RuntimeContext:
    return RuntimeContext(cuda_mode=True, device="cuda:0", use_nvdec=False,
                          default_runtime="cuda", tensor_on_cuda=True)
```

CPU helpers work without change because `tensor_on_cuda` defaults to `False`.

---

### Correction 2 — Task 5: Fix `_run_direct` CUDA path gate + fix `_load_yolo`

**Fix 1 — `_run_direct` must gate on `tensor_on_cuda`, not `cuda_mode`**

The current check `if runtime.cuda_mode:` routes `onnx_cuda` and `tensorrt` through `_extract_raw_tensors`, but those models return CPU numpy, not CUDA tensors. Change the gate:

```python
def _run_direct(
    frames: list,
    model: Any,
    config: OBBConfig,
    runtime: RuntimeContext,
) -> list[OBBResult | _RawOBBTensors]:
    conf_floor = config.direct.confidence_floor if config.direct else 1e-3
    results = model.predict(
        frames, conf=conf_floor, iou=1.0, verbose=False, device=runtime.device,
    )
    # Only native PyTorch "cuda" runtime leaves tensors on device.
    # onnx_cuda and tensorrt: predict() returns CPU numpy regardless of GPU use.
    if runtime.tensor_on_cuda:
        return [_extract_raw_tensors(r, idx) for idx, r in enumerate(results)]
    return [_extract_obb_result(r, idx) for idx, r in enumerate(results)]
```

**Fix 2 — `_load_yolo` must not call `.to()` on ONNX/TRT models**

`YOLO.to(device)` only works for native `.pt` PyTorch models. For ONNX (`.onnx`) and TensorRT (`.engine`) artifacts, the execution provider is selected at `predict()` time via the `device=` parameter — `.to()` silently does nothing or raises. CoreML (`onnx_coreml`) is handled by Ultralytics when `device="mps"` is passed to `predict()`.

```python
def _load_yolo(model_path: str, compute_runtime: ComputeRuntime) -> Any:
    from ultralytics import YOLO
    model = YOLO(model_path)
    # Only native PyTorch models support .to(). ONNX and TensorRT artifacts
    # (.onnx, .engine) ignore .to() — their execution provider is set at
    # predict() time via device=. CoreML provider activates when device="mps".
    if compute_runtime == "cuda":
        model.to("cuda:0")
    elif compute_runtime == "mps":
        model.to("mps")
    # cpu: already on CPU by default
    # onnx_cuda, onnx_cpu, onnx_coreml, tensorrt: device flows through predict()
    return model
```

Also update the OBB test to add a test for this:

```python
def test_load_yolo_does_not_call_to_for_onnx(monkeypatch):
    """ONNX models must not have .to() called — it silently does nothing and
    misleads readers. Device is set at predict() time."""
    calls = []
    class FakeYOLO:
        def to(self, device):
            calls.append(device)
            return self
        def predict(self, *a, **kw):
            return []
    monkeypatch.setattr("hydra_suite.core.inference.stages.obb.YOLO", lambda p: FakeYOLO())
    from hydra_suite.core.inference.stages.obb import _load_yolo
    _load_yolo("/m.onnx", "onnx_cuda")
    assert len(calls) == 0   # .to() must not have been called
    _load_yolo("/m.engine", "tensorrt")
    assert len(calls) == 0
    _load_yolo("/m.onnx", "onnx_coreml")
    assert len(calls) == 0
```

---

### Correction 3 — Task 7: Fix `_extract_canonical_gpu` theta matrix + batch kernel calls + gate on `tensor_on_cuda`

Three sub-fixes for `stages/crops.py`.

**Fix 1 — Wrong theta matrix for non-square crops**

In `grid_sample`'s normalised coordinate frame, the output x-axis spans `[-1, 1]` over `out_w` pixels and the y-axis over `out_h` pixels. The affine theta `[[a, b, tx], [c, d, ty]]` maps *output normalised coords* to *input normalised coords*. The correct scale factors are:

- `theta[0, 0]` (out-x → in-x): `cos * (out_w / W)` — scaling the x-output by x-input ratio
- `theta[0, 1]` (out-y → in-x): `-sin * (out_h / W)` — out-y spans `out_h` but maps to x-input; divide by W
- `theta[1, 0]` (out-x → in-y): `sin * (out_w / H)` — out-x spans `out_w` but maps to y-input; divide by H
- `theta[1, 1]` (out-y → in-y): `cos * (out_h / H)` — scaling the y-output by y-input ratio

**Fix 2 — Batch all per-detection affine_grid + grid_sample calls into one**

Building a theta matrix per detection and calling `affine_grid + grid_sample` N times = 2N GPU kernel launches. All detections in one frame use the same source tensor so we can batch: compute one fixed output size, stack all theta matrices to `(N, 2, 3)`, call `affine_grid` once to get `(N, out_h, out_w, 2)`, call `grid_sample` once over the replicated source.

**Fix 3 — Gate on `tensor_on_cuda`, not `cuda_mode`**

`onnx_cuda` runs on GPU but returns CPU numpy from the model. Uploading the frame to GPU and extracting crops on GPU only to immediately `.cpu().numpy()` them for the ONNX model is wasteful. Only use `_extract_canonical_gpu` when `runtime.tensor_on_cuda` is True (native PyTorch `cuda`).

**Corrected `stages/crops.py` (replaces Task 7 Step 3)**:

```python
from __future__ import annotations

import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..result import OBBResult
from ..runtime import RuntimeContext


def extract_canonical_crops(
    frame: np.ndarray | torch.Tensor,
    obb_result: OBBResult,
    canonical_aspect_ratio: float,
    canonical_margin: float,
    runtime: RuntimeContext,
) -> torch.Tensor:
    """Extract OBB-aligned canonical crops. Returns (N, C, H, W) tensor on runtime.device.

    GPU path (tensor_on_cuda only): single batched affine_grid + grid_sample call.
    CPU path: cv2.warpAffine per crop → stacked CPU tensor.
    onnx_cuda/tensorrt use CPU path even though cuda_mode=True — their downstream
    models take CPU numpy, so GPU crop upload+download would be pure waste.
    """
    n = obb_result.num_detections
    if n == 0:
        return torch.zeros((0, 3, 64, 64), dtype=torch.float32)

    if runtime.tensor_on_cuda:
        return _extract_canonical_gpu(frame, obb_result, canonical_aspect_ratio,
                                      canonical_margin, runtime.device)
    return _extract_canonical_cpu(frame, obb_result, canonical_aspect_ratio,
                                  canonical_margin)


def extract_aabb_crops(
    frame: np.ndarray,
    obb_result: OBBResult,
    padding: float,
) -> list[np.ndarray]:
    """Extract axis-aligned bounding box crops for AprilTag detection.
    Always CPU numpy. frame must be a numpy array (already .cpu().numpy() on CUDA path)."""
    if obb_result.num_detections == 0:
        return []
    h, w = frame.shape[:2]
    crops: list[np.ndarray] = []
    for i in range(obb_result.num_detections):
        corners = obb_result.corners[i]   # (4, 2)
        x1, y1 = corners[:, 0].min(), corners[:, 1].min()
        x2, y2 = corners[:, 0].max(), corners[:, 1].max()
        bw, bh = x2 - x1, y2 - y1
        pad = padding * max(bw, bh)
        ox1 = max(0, int(x1 - pad))
        oy1 = max(0, int(y1 - pad))
        ox2 = min(w, int(x2 + pad))
        oy2 = min(h, int(y2 + pad))
        crop = frame[oy1:oy2, ox1:ox2]
        crops.append(crop if crop.size > 0 else np.zeros((1, 1, 3), dtype=np.uint8))
    return crops


# ── CPU canonical crop extraction ─────────────────────────────────────────────

def _extract_canonical_cpu(
    frame: np.ndarray | torch.Tensor,
    obb: OBBResult,
    aspect_ratio: float,
    margin: float,
) -> torch.Tensor:
    if isinstance(frame, torch.Tensor):
        arr = frame.cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
    else:
        arr = frame

    crops: list[np.ndarray] = []
    for i in range(obb.num_detections):
        crop = _warp_canonical_crop(arr, obb.centroids[i], obb.angles[i],
                                    obb.sizes[i], aspect_ratio, margin)
        crops.append(crop)

    stacked = np.stack(crops, axis=0)           # (N, H, W, C)
    t = torch.from_numpy(stacked).permute(0, 3, 1, 2).float() / 255.0
    return t


def _warp_canonical_crop(
    frame: np.ndarray,
    centroid: np.ndarray,
    angle: float,
    size: float,
    aspect_ratio: float,
    margin: float,
) -> np.ndarray:
    """Extract a rotated crop centred on centroid, aligned so OBB is upright."""
    side = math.sqrt(size) * margin
    out_w = max(int(side * aspect_ratio), 4)
    out_h = max(int(side), 4)

    cx, cy = float(centroid[0]), float(centroid[1])
    angle_deg = float(np.degrees(angle))

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    M[0, 2] += out_w / 2 - cx
    M[1, 2] += out_h / 2 - cy

    crop = cv2.warpAffine(frame, M, (out_w, out_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return crop


# ── GPU canonical crop extraction ─────────────────────────────────────────────

def _extract_canonical_gpu(
    frame: torch.Tensor | np.ndarray,
    obb: OBBResult,
    aspect_ratio: float,
    margin: float,
    device: str,
) -> torch.Tensor:
    """Batched affine crop extraction on CUDA tensor via a single grid_sample call.

    All N crops are extracted in one affine_grid + grid_sample kernel pair.
    Output size is fixed to the largest canonical crop in the batch — smaller
    crops are slightly over-padded, which is acceptable for downstream models.
    """
    if isinstance(frame, np.ndarray):
        if frame.ndim == 3:
            frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
        frame = frame.to(device)

    if frame.ndim == 3:
        frame = frame.unsqueeze(0)   # (1, C, H, W)

    _, C, H, W = frame.shape
    n = obb.num_detections

    # Compute per-detection canonical output sizes
    sides = [math.sqrt(float(obb.sizes[i])) * margin for i in range(n)]
    out_ws = [max(int(s * aspect_ratio), 4) for s in sides]
    out_hs = [max(int(s), 4) for s in sides]
    # Fixed output size: use maximum so all crops can be batched in one call
    out_w = max(out_ws)
    out_h = max(out_hs)

    # Build theta matrices for all N detections: shape (N, 2, 3)
    # affine_grid maps output normalised coords → input normalised coords.
    # theta[row, col] meaning (output pixel space → input pixel space):
    #   [0,0]: cos * (out_w / W)   — x-output direction scaling by x-input extent
    #   [0,1]: -sin * (out_h / W)  — y-output direction rotated into x-input; scale by W
    #   [0,2]: 2*cx/W - 1          — normalised centre x
    #   [1,0]: sin * (out_w / H)   — x-output direction rotated into y-input; scale by H
    #   [1,1]: cos * (out_h / H)   — y-output direction scaling by y-input extent
    #   [1,2]: 2*cy/H - 1          — normalised centre y
    thetas = []
    for i in range(n):
        cx = float(obb.centroids[i, 0])
        cy = float(obb.centroids[i, 1])
        angle = float(obb.angles[i])
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)
        ncx = 2.0 * cx / W - 1.0
        ncy = 2.0 * cy / H - 1.0
        thetas.append([
            [cos_a * (out_w / W), -sin_a * (out_h / W), ncx],
            [sin_a * (out_w / H),  cos_a * (out_h / H), ncy],
        ])

    theta_t = torch.tensor(thetas, dtype=torch.float32, device=device)  # (N, 2, 3)

    # Replicate source frame N times for batched grid_sample
    frame_batch = frame.expand(n, -1, -1, -1)  # (N, C, H, W) — no memory copy

    grid = F.affine_grid(theta_t, (n, C, out_h, out_w), align_corners=False)
    crops = F.grid_sample(frame_batch, grid, mode="bilinear",
                          padding_mode="zeros", align_corners=False)
    return crops   # (N, C, out_h, out_w)
```

---

### Correction 4 — Task 14: Fix all `runner.py` bugs

The runner.py block in Task 14 has seven bugs. Apply all seven replacements to that code block before implementation.

**Bug 4 — `config.cnn` should be `config.cnn_phases`**

`InferenceConfig` defines `cnn_phases: list[CNNConfig]`, not `cnn`. Every occurrence of `config.cnn` in `_load_all_models`, `_open_caches`, `run_realtime`, `_run_batch`, and `load_frame` will raise `AttributeError`.

**Bug 5 — `.enabled` checks on optional sub-configs**

`config.headtail`, `config.pose`, and individual `CNNConfig` entries have no `.enabled` field. The presence of `config.headtail` or `config.pose` is the flag.

**Bug 6 — Wrong `extract_canonical_crops` call signature**

`extract_canonical_crops(frame, obb, config, runtime)` should be
`extract_canonical_crops(frame, obb, aspect_ratio, margin, runtime)`.

**Bug 7 — `filtered_obb.centers` → `filtered_obb.num_detections`**

**Bug 8 — `ThreadPoolExecutor` inside the per-frame loop**

**Bug 9 — No cross-frame batching for HeadTail/CNN/Pose**

**Bug 12 — `config.obb.batch_size` → `config.detection_batch_size`**

**Corrected `_load_all_models` (replaces the function in Task 14 Step 5)**:

```python
def _load_all_models(config: InferenceConfig, runtime: RuntimeContext) -> _AllModels:
    from .stages.obb import load_obb_models
    from .stages.headtail import load_headtail_model
    from .stages.cnn import load_cnn_model
    from .stages.pose import load_pose_model
    from .stages.apriltag import load_apriltag_model

    obb = load_obb_models(config.obb, runtime)
    headtail = load_headtail_model(config.headtail, runtime) if config.headtail is not None else None
    cnn = [load_cnn_model(c, runtime) for c in config.cnn_phases]
    pose = load_pose_model(config.pose, runtime) if config.pose is not None else None
    apriltag = load_apriltag_model(config.apriltag) if config.apriltag.enabled else None
    return _AllModels(obb=obb, headtail=headtail, cnn=cnn, pose=pose, apriltag=apriltag)
```

**Corrected `_open_caches` (replaces the function in Task 14 Step 5)**:

```python
def _open_caches(config: InferenceConfig, cache_dir: Path) -> _CacheSet:
    return _CacheSet(
        detection=DetectionCacheHandle(
            cache_dir / "detection.npz",
            detection_cache_key(config.obb),
        ),
        headtail=HeadTailCacheHandle(
            cache_dir / "headtail.npz",
            headtail_cache_key(config.headtail),
        ) if config.headtail is not None else None,
        cnn=[
            CNNCacheHandle(cache_dir / f"cnn_{c.label}.npz", cnn_cache_key(c))
            for c in config.cnn_phases
        ],
        pose=PoseCacheHandle(
            cache_dir / "pose.npz",
            pose_cache_key(config.pose),
        ) if config.pose is not None else None,
        apriltag=AprilTagCacheHandle(
            cache_dir / "apriltag.npz",
            apriltag_cache_key(config.apriltag),
        ) if config.apriltag.enabled else None,
    )
```

**Corrected `run_realtime` (replaces in Task 14 Step 5)**:

```python
    def run_realtime(
        self,
        frame: np.ndarray,
        roi_mask: np.ndarray | None = None,
        roi_mask_cuda: Any = None,
    ) -> FrameResult:
        """Run full inference on a single frame. No cache I/O."""
        raw_list = run_obb([frame], self.config.obb, self._models.obb, self.runtime)
        raw = raw_list[0]
        filtered_obb, det_indices = filter_with_indices(
            filter_raw(raw, self.config.obb, roi_mask, roi_mask_cuda, self.runtime)
            if isinstance(raw, _RawOBBTensors)
            else raw,
            self.config.obb,
            roi_mask,
        )

        if filtered_obb.num_detections == 0:
            return _build_frame_result(0, filtered_obb, np.zeros(0, np.int32),
                                       None, [], None, None)

        ar = self.config.headtail.canonical_aspect_ratio if self.config.headtail else 2.0
        mg = self.config.headtail.canonical_margin if self.config.headtail else 1.3
        canonical_crops = extract_canonical_crops(frame, filtered_obb, ar, mg, self.runtime)

        aabb_crops = (
            extract_aabb_crops(frame, filtered_obb, padding=self.config.apriltag.crop_padding)
            if self._models.apriltag else []
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            ht_fut = (
                pool.submit(run_headtail, canonical_crops, self.config.headtail,
                            self._models.headtail, self.runtime)
                if self._models.headtail else None
            )
            cnn_futs = [
                pool.submit(run_cnn, canonical_crops, cfg, mdl, self.runtime)
                for cfg, mdl in zip(self.config.cnn_phases, self._models.cnn)
            ]
            pose_fut = (
                pool.submit(run_pose, canonical_crops, self.config.pose,
                            self._models.pose, self.runtime)
                if self._models.pose else None
            )
            at_fut = (
                pool.submit(run_apriltag, aabb_crops, self.config.apriltag,
                            self._models.apriltag)
                if self._models.apriltag else None
            )
            ht_result = ht_fut.result() if ht_fut else None
            cnn_results = [f.result() for f in cnn_futs]
            pose_result = pose_fut.result() if pose_fut else None
            at_result = at_fut.result() if at_fut else None

        return _build_frame_result(0, filtered_obb, det_indices,
                                   ht_result, cnn_results, pose_result, at_result)
```

**Corrected `run_batch_pass` (fixes Bug 12 — wrong batch_size field)**:

```python
    def run_batch_pass(self, video_path: Path, progress_cb=None) -> None:
        import cv2

        if self.cache_dir is None:
            raise RuntimeError("cache_dir must be set before calling run_batch_pass")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        caches = _open_caches(self.config, self.cache_dir)
        self._caches = caches
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        batch_size = self.config.detection_batch_size  # fixed: InferenceConfig field, not OBBConfig

        frames_buf: list[np.ndarray] = []
        indices_buf: list[int] = []
        processed = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_buf.append(frame)
                indices_buf.append(processed)
                processed += 1
                if len(frames_buf) == batch_size:
                    self._run_batch(frames_buf, indices_buf, caches)
                    frames_buf.clear()
                    indices_buf.clear()
                if progress_cb and total_frames > 0 and processed % max(1, total_frames // 100) == 0:
                    progress_cb(processed, total_frames)
            if frames_buf:
                self._run_batch(frames_buf, indices_buf, caches)
            if progress_cb:
                progress_cb(processed, total_frames)
        finally:
            cap.release()
            for h in caches.all_handles():
                h.close()
```

**Corrected `_run_batch` — fixes Bugs 5, 6, 7, 8, 9 (cross-frame batching, thread pool, call signatures)**:

```python
    def _run_batch(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
        caches: _CacheSet,
    ) -> None:
        """Run OBB + identity on one batch of frames.

        OBB detection runs on all frames together (already batched by run_obb).
        All crops from all frames are collected first, then identity models (HeadTail,
        CNN, Pose) run once over the combined crop tensor — cross-frame batching
        maximises GPU utilisation and reduces kernel launch overhead.
        ThreadPoolExecutor is created ONCE per batch call, not per frame.
        """
        import torch

        ar = self.config.headtail.canonical_aspect_ratio if self.config.headtail else 2.0
        mg = self.config.headtail.canonical_margin if self.config.headtail else 1.3

        raw_list = run_obb(frames, self.config.obb, self._models.obb, self.runtime)

        # Materialize, filter, and extract crops for all frames first
        all_crops: list[torch.Tensor] = []
        frame_data: list = []  # (frame_idx, frame, filtered_obb, det_indices, crop_slice)
        crop_offset = 0

        for frame, frame_idx, raw in zip(frames, frame_indices, raw_list):
            if isinstance(raw, _RawOBBTensors):
                obb_result = materialize_tensors(raw)
            else:
                obb_result = raw

            caches.detection.write_frame(frame_idx, obb=obb_result)

            filtered_obb, det_indices = filter_with_indices(obb_result, self.config.obb)
            if filtered_obb.num_detections == 0:
                frame_data.append((frame_idx, frame, filtered_obb, det_indices, slice(0, 0)))
                continue

            crops = extract_canonical_crops(frame, filtered_obb, ar, mg, self.runtime)
            n_crops = crops.shape[0]
            all_crops.append(crops)
            frame_data.append(
                (frame_idx, frame, filtered_obb, det_indices,
                 slice(crop_offset, crop_offset + n_crops))
            )
            crop_offset += n_crops

        if not all_crops:
            return

        # Cross-frame batched identity inference — one call for all crops from all frames
        batched_crops = torch.cat(all_crops, dim=0)  # (total_crops, C, H, W)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            ht_fut = (
                pool.submit(run_headtail, batched_crops, self.config.headtail,
                            self._models.headtail, self.runtime)
                if self._models.headtail else None
            )
            cnn_futs = [
                pool.submit(run_cnn, batched_crops, cfg, mdl, self.runtime)
                for cfg, mdl in zip(self.config.cnn_phases, self._models.cnn)
            ]
            pose_fut = (
                pool.submit(run_pose, batched_crops, self.config.pose,
                            self._models.pose, self.runtime)
                if self._models.pose else None
            )
            ht_all = ht_fut.result() if ht_fut else None
            cnn_all = [f.result() for f in cnn_futs]
            pose_all = pose_fut.result() if pose_fut else None

        # Scatter results back to per-frame cache writes
        for frame_idx, frame, filtered_obb, det_indices, crop_sl in frame_data:
            if det_indices.size == 0:
                continue

            ht_frame = _slice_headtail(ht_all, crop_sl)
            cnn_frame = [_slice_cnn(r, crop_sl) for r in cnn_all]
            pose_frame = _slice_pose(pose_all, crop_sl)

            if caches.headtail and ht_frame is not None:
                caches.headtail.write_frame(frame_idx, det_indices=det_indices, headtail=ht_frame)
            for cache, result in zip(caches.cnn, cnn_frame):
                if result is not None:
                    cache.write_frame(frame_idx, det_indices=det_indices, cnn=result)
            if caches.pose and pose_frame is not None:
                caches.pose.write_frame(frame_idx, det_indices=det_indices, pose=pose_frame)

            if self._models.apriltag:
                aabb_crops = extract_aabb_crops(
                    frame, filtered_obb, padding=self.config.apriltag.crop_padding
                )
                at_result = run_apriltag(aabb_crops, self.config.apriltag, self._models.apriltag)
                if caches.apriltag and at_result is not None:
                    caches.apriltag.write_frame(frame_idx, result=at_result)
```

**Add three slice helpers to `runner.py`** (called by `_run_batch` above):

```python
def _slice_headtail(ht: "HeadTailResult | None", sl: slice) -> "HeadTailResult | None":
    if ht is None or sl.start == sl.stop:
        return None
    from .result import HeadTailResult
    import numpy as np
    return HeadTailResult(
        heading_hints=ht.heading_hints[sl],
        heading_confidences=ht.heading_confidences[sl],
        directed_mask=ht.directed_mask[sl],
        canonical_affines=ht.canonical_affines[sl] if ht.canonical_affines is not None else None,
    )


def _slice_cnn(cnn: "CNNResult | None", sl: slice) -> "CNNResult | None":
    if cnn is None or sl.start == sl.stop:
        return None
    from .result import CNNResult
    return CNNResult(label=cnn.label, predictions=cnn.predictions[sl.start:sl.stop])


def _slice_pose(pose: "PoseResult | None", sl: slice) -> "PoseResult | None":
    if pose is None or sl.start == sl.stop:
        return None
    from .result import PoseResult
    import numpy as np
    return PoseResult(keypoints=pose.keypoints[sl], valid_mask=pose.valid_mask[sl])
```

---

### Correction 5 — Issue 13: Producer-consumer pipeline for sustained GPU throughput

**Read this section before implementing `run_batch_pass` in Task 15.**

The sequential loop in `run_batch_pass` leaves the GPU idle while video frames are read from disk and while cache is written. On a typical 90-minute video, the GPU is idle 30–50% of the time.

The fix is a two-thread pipeline:

- **Reader thread**: reads frames from disk, writes `(frame_idx, frame)` tuples into a bounded queue.
- **Worker thread** (main): drains batches from the reader queue, runs OBB + identity + cache write.

This overlaps disk I/O with GPU compute. OBB inference on batch N+1 runs while the reader fills the buffer for batch N+2.

**Replace `run_batch_pass` in Task 15 with this pipelined version**:

```python
    def run_batch_pass(self, video_path: Path, progress_cb=None) -> None:
        """Run inference on every frame of `video_path` and write results to cache.

        Uses a producer-consumer pair: a reader thread prefetches frames from disk
        into a bounded queue while the main thread runs OBB + identity on each batch.
        This overlaps I/O with GPU compute for sustained throughput on long videos.
        """
        import cv2
        import queue
        import threading

        if self.cache_dir is None:
            raise RuntimeError("cache_dir must be set before calling run_batch_pass")

        # Probe video before opening caches so IOError is raised early.
        cap_probe = cv2.VideoCapture(str(video_path))
        if not cap_probe.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_probe.release()

        caches = _open_caches(self.config, self.cache_dir)
        self._caches = caches

        batch_size = self.config.detection_batch_size
        # Queue holds at most 2 full batches pre-fetched ahead of the worker.
        frame_q: queue.Queue = queue.Queue(maxsize=2 * batch_size)
        _DONE = object()  # sentinel

        def _reader() -> None:
            cap = cv2.VideoCapture(str(video_path))
            try:
                idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_q.put((idx, frame))
                    idx += 1
            finally:
                cap.release()
                frame_q.put(_DONE)

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        processed = 0
        frames_buf: list[np.ndarray] = []
        indices_buf: list[int] = []

        try:
            while True:
                item = frame_q.get()
                if item is _DONE:
                    break
                frame_idx, frame = item
                frames_buf.append(frame)
                indices_buf.append(frame_idx)
                processed += 1

                if len(frames_buf) == batch_size:
                    self._run_batch(frames_buf, indices_buf, caches)
                    frames_buf.clear()
                    indices_buf.clear()

                if progress_cb and total_frames > 0 and processed % max(1, total_frames // 100) == 0:
                    progress_cb(processed, total_frames)

            if frames_buf:
                self._run_batch(frames_buf, indices_buf, caches)
            if progress_cb:
                progress_cb(processed, total_frames)

        finally:
            # Drain queue so reader thread can exit if it blocked on put()
            while not frame_q.empty():
                try:
                    frame_q.get_nowait()
                except queue.Empty:
                    break
            reader.join(timeout=5.0)
            for h in caches.all_handles():
                h.close()
```

**Add one test** to `tests/test_inference_runner_batch.py`:

```python
def test_run_batch_pass_reader_thread_drains_queue(tmp_path):
    """Reader thread puts frames into queue; worker drains them in batches."""
    import queue as _queue
    import threading

    from hydra_suite.core.inference.runner import InferenceRunner
    cfg_mock = MagicMock()
    cfg_mock.detection_batch_size = 2
    cfg_mock.headtail = None
    cfg_mock.cnn_phases = []
    cfg_mock.pose = None
    cfg_mock.apriltag = MagicMock(enabled=False)

    frame_q: _queue.Queue = _queue.Queue(maxsize=10)
    _DONE = object()

    batches_processed = []

    with patch("hydra_suite.core.inference.runner._load_all_models"):
        with patch("hydra_suite.core.inference.runner._open_caches") as mock_open:
            mock_caches = MagicMock()
            mock_caches.all_handles.return_value = []
            mock_open.return_value = mock_caches

            runner = InferenceRunner.__new__(InferenceRunner)
            runner.config = cfg_mock
            runner.cache_dir = tmp_path
            runner._caches = None
            runner._models = MagicMock()

            original_run_batch = runner._run_batch
            def recording_run_batch(frames, indices, caches):
                batches_processed.append(len(frames))
            runner._run_batch = recording_run_batch

            # Patch VideoCapture to deliver 5 synthetic frames then stop
            fake_frame = np.zeros((4, 4, 3), dtype=np.uint8)
            call_count = 0
            class FakeCap:
                def isOpened(self): return True
                def get(self, prop): return 5.0
                def read(self):
                    nonlocal call_count
                    call_count += 1
                    if call_count <= 5:
                        return True, fake_frame
                    return False, None
                def release(self): pass

            with patch("cv2.VideoCapture", return_value=FakeCap()):
                runner.run_batch_pass(tmp_path / "v.mp4")

    assert sum(batches_processed) == 5      # all 5 frames processed
    assert len(batches_processed) == 3      # ceil(5 / 2) = 3 batches
```

---

*All 13 correctness and performance issues are now documented. Implement Phases 1–5 using the corrected code in this section and in the "Type Consistency Corrections" section wherever they conflict with the phase task steps above.*

---

## Downstream Compatibility Corrections (2026-05-04)

**Read this section before implementing any task.** A downstream-consumer audit found that the original plan, executed verbatim, would crash on import the moment deletion happens — five kept files (`optimizer.py`, `optimizer_workers.py`, `properties/export.py`, `dataset/oriented_video.py`, `posekit/gui/workers.py`) and seven `__init__.py` re-exports import from deleted modules at top level, and no feature flag protects against `ImportError`. The audit also found data-flow gaps: `detection_ids` is missing from `OBBResult`, the online identity decoder needs full probability vectors, and several "kept as-is" files actually need targeted rewires.

The corrections below add new tasks (17b–17g, 18a) and modify existing ones. Apply them in numerical order. Where a correction modifies an existing task, treat it as superseding the original code in that task.

---

### Correction 14 — Task 4: Add `detection_ids` to `OBBResult` and propagate through stages

The legacy worker uses an integer key `frame_idx * 10000 + slot` to join detections to CSV rows, identity evidence, pose keypoint maps, and AprilTag observations. `OBBResult` as originally specified has no equivalent field. Add it.

**Modify the `OBBResult` dataclass in Task 4** (`src/hydra_suite/core/inference/result.py`):

```python
DETECTION_ID_STRIDE = 10000   # max detections per frame; matches legacy stride


@dataclass
class OBBResult:
    frame_idx: int
    centroids: np.ndarray        # (D, 2) cx, cy
    angles: np.ndarray           # (D,) radians
    sizes: np.ndarray            # (D,) area px²
    shapes: np.ndarray           # (D, 2) ellipse_area, aspect_ratio
    confidences: np.ndarray      # (D,) raw detection confidence
    corners: np.ndarray          # (D, 4, 2) OBB corners
    detection_ids: np.ndarray    # (D,) int64; primary key for downstream consumers

    @property
    def num_detections(self) -> int:
        return int(len(self.confidences))

    @staticmethod
    def make_detection_ids(frame_idx: int, num_detections: int) -> np.ndarray:
        """Generate the legacy-compatible primary keys: frame_idx * STRIDE + slot."""
        return (
            np.arange(num_detections, dtype=np.int64)
            + np.int64(frame_idx) * np.int64(DETECTION_ID_STRIDE)
        )
```

**Add `detection_ids` to every `OBBResult` construction site:**

- `stages/obb.py` (`run_obb`): generate fresh IDs after raw detections are produced, BEFORE any filtering. Filtering preserves the original IDs of surviving detections.
- `stages/filtering.py` (`filter_detections`, `filter_with_indices`): subset `detection_ids` by the same indices used to subset `centroids`/`confidences`/etc. Never re-generate IDs after filtering — IDs are stable across forward/backward passes by design.
- `cache/detection.py`: write `detection_ids` to the npz alongside `centroids`/`angles`/etc. Read back with the same dtype.
- All test helpers (`_make_obb` in `tests/test_inference_result.py`, Phase 4-5 test helpers): add `detection_ids=OBBResult.make_detection_ids(0, n)`.

**Add a test** to `tests/test_inference_result.py`:

```python
def test_detection_ids_are_unique_across_frames():
    ids_f0 = OBBResult.make_detection_ids(0, 5)
    ids_f1 = OBBResult.make_detection_ids(1, 5)
    ids_f100 = OBBResult.make_detection_ids(100, 3)
    assert set(ids_f0).isdisjoint(set(ids_f1))
    assert set(ids_f0).isdisjoint(set(ids_f100))
    assert ids_f0.dtype == np.int64
    # Stride preserved: id 0 of frame 1 = STRIDE
    from hydra_suite.core.inference.result import DETECTION_ID_STRIDE
    assert ids_f1[0] == DETECTION_ID_STRIDE


def test_detection_ids_survive_filtering():
    """filter_with_indices must subset detection_ids, not regenerate them."""
    from hydra_suite.core.inference.filtering import filter_with_indices
    from hydra_suite.core.inference.config import OBBConfig
    raw = OBBResult(
        frame_idx=7,
        centroids=np.zeros((4, 2)),
        angles=np.zeros(4),
        sizes=np.ones(4) * 100,
        shapes=np.ones((4, 2)),
        confidences=np.array([0.9, 0.1, 0.8, 0.2]),
        corners=np.zeros((4, 4, 2)),
        detection_ids=OBBResult.make_detection_ids(7, 4),
    )
    cfg = OBBConfig(confidence_threshold=0.5, min_object_size=0, nms_iou_threshold=1.0)
    filtered, indices = filter_with_indices(raw, cfg)
    # Surviving IDs should be raw.detection_ids[indices], not freshly minted
    np.testing.assert_array_equal(filtered.detection_ids, raw.detection_ids[indices])
```

---

### Correction 15 — Task 4: Make `HeadTailResult.canonical_affines` Optional

The cache-loaded path cannot reconstruct `canonical_affines` (it stores model outputs only). Use `None` to signal "not available; recompute from OBB if needed."

**Modify the `HeadTailResult` dataclass in Task 4:**

```python
@dataclass
class HeadTailResult:
    heading_hints: np.ndarray              # (D,) radians; nan = no confident direction
    heading_confidences: np.ndarray        # (D,)
    directed_mask: np.ndarray              # (D,) uint8; 1 = heading trusted
    canonical_affines: np.ndarray | None   # (D, 2, 3); None when loaded from cache
```

**Update every consumer that reads `headtail.canonical_affines`** to check for `None` first. There is exactly one consumer in tracking code (visualization debug overlay in `core/tracking/visualization.py`); it falls back to `np.eye(2, 3)` when `None`.

This supersedes "Type Consistency Corrections" → "HeadTailResult optional canonical_affines", which used a zeros sentinel. The `None` sentinel is preferred — it makes "no affines available" explicit rather than silently passing zero matrices into downstream math.

---

### Correction 16 — Task 12: Add `CACHE_SCHEMA_VERSION` to `CacheKey`

Without a schema version field, the legacy `CNNIdentityCache` (which stored *post-calibration* probabilities) can be silently re-used by the new pipeline (which expects *raw* pre-calibration probabilities), corrupting all identity decoding during the feature-flag coexistence window.

**Add to `src/hydra_suite/core/inference/cache/base.py` in Task 12:**

```python
# Bump this whenever the on-disk schema of any cached result type changes:
# - Adding/removing/renaming fields
# - Changing dtype or shape conventions
# - Changing whether the cache stores raw vs calibrated outputs
# v1 = legacy pre-redesign caches (DetectionCache, CNNIdentityCache, etc.)
# v2 = new pipeline (this redesign)
CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class CacheKey:
    schema_version: int                                # CACHE_SCHEMA_VERSION at write time
    model_path: str
    model_mtime: float
    config_hash: str

    def matches(self, other: "CacheKey") -> bool:
        return (
            self.schema_version == other.schema_version
            and self.model_path == other.model_path
            and abs(self.model_mtime - other.model_mtime) < 1e-3
            and self.config_hash == other.config_hash
        )
```

**Modify `CacheHandle.is_valid()`** to short-circuit on schema mismatch:

```python
class CacheHandle(ABC):
    def is_valid(self) -> bool:
        try:
            stored_key = self._read_key_from_disk()
        except (FileNotFoundError, OSError, KeyError):
            return False
        # Schema mismatch invalidates regardless of model identity.
        if stored_key.schema_version != CACHE_SCHEMA_VERSION:
            return False
        return stored_key.matches(self._expected_key)
```

**Add a test** to `tests/test_inference_cache.py`:

```python
def test_cache_invalidated_on_schema_version_mismatch(tmp_path):
    """A cache written with an older schema_version must be detected as invalid."""
    from hydra_suite.core.inference.cache.base import CacheKey, CACHE_SCHEMA_VERSION
    from hydra_suite.core.inference.cache.detection import DetectionCacheHandle

    # Write a v1 cache (simulating legacy)
    legacy_key = CacheKey(
        schema_version=CACHE_SCHEMA_VERSION - 1,
        model_path="/fake/yolo.pt",
        model_mtime=12345.0,
        config_hash="abc",
    )
    handle = DetectionCacheHandle(tmp_path / "det.npz", expected_key=legacy_key)
    handle._write_key_to_disk(legacy_key)

    # Open with current schema_version → must invalidate
    current_key = CacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        model_path="/fake/yolo.pt",
        model_mtime=12345.0,
        config_hash="abc",
    )
    handle2 = DetectionCacheHandle(tmp_path / "det.npz", expected_key=current_key)
    assert handle2.is_valid() is False
```

---

### Correction 17 — Task 16: Identity decoder reads `cnn_factors[i].calibrated_probabilities`, NOT `resolved_*`

The original plan implies `OnlineIdentityDecoder.update_frame()` consumes `DetectionIdentityEvidence.resolved_label` / `resolved_confidence`. Those are top-1 convenience fields for the CSV writer and visualization. The decoder needs the **full** posterior distribution to do catalog remapping — the legacy code accesses raw multi-class probability vectors and reshapes them into the catalog's class space.

**Document the contract explicitly** in Task 16 Step 5 (`worker.py` integration). After the loop that builds `FrameIdentityEvidence`, the dispatch to the decoder must look like:

```python
# Online decoder consumes the full posterior, not the top-1.
for det_evidence in frame_evidence.detections:
    track_id = self._track_id_for_detection(det_evidence.det_index)
    if track_id is None:
        continue

    if det_evidence.is_authoritative:
        # AprilTag: pass through as one-hot evidence over the catalog.
        self._online_decoder.update_apriltag(
            track_id=track_id,
            tag_id=det_evidence.apriltag_tag_id,
            label=det_evidence.apriltag_label,
            frame_idx=frame_evidence.frame_idx,
        )
    elif det_evidence.cnn_factors:
        # CNN: pass the full calibrated distribution. The decoder remaps it
        # into the catalog space inside _remap_source_log_probs_to_catalog().
        for factor in det_evidence.cnn_factors:
            self._online_decoder.update_cnn(
                track_id=track_id,
                factor_name=factor.factor_name,
                class_names=factor.class_names,
                calibrated_probabilities=factor.calibrated_probabilities,
                frame_idx=frame_evidence.frame_idx,
            )
```

The `update_cnn` and `update_apriltag` methods exist in `core/identity/online.py` today (kept as-is) — the legacy worker calls them via slightly different argument names. This correction documents the new call shape and ensures `class_names` is forwarded so the decoder can do its own catalog remap (avoiding a re-derivation inside the builder).

**Add a test** to `tests/test_worker_inference_integration.py`:

```python
def test_online_decoder_receives_full_calibrated_distribution(tmp_path):
    """OnlineIdentityDecoder.update_cnn must be called with the full probability vector,
    not just the top-1 label and confidence."""
    # ... build a FrameResult with one detection whose CNNFactorPrediction has
    # raw_probabilities = [0.1, 0.2, 0.7]
    # ... patch worker._online_decoder.update_cnn and assert call kwargs include
    # calibrated_probabilities of length 3 (not a scalar confidence)
```

---

### Correction 18 — New Task 17b: Rewire `core/identity/pose/features.py` to read from new `PoseCache`

`features.py` is "kept as-is" but currently calls `pose_props_cache.get_frame(frame_idx)` and reads `frame["detection_ids"]` and `frame["pose_keypoints"]` — both come from the deleted `IndividualPropertiesCache`. The new `PoseCache` returns a `PoseResult` dataclass with different shape.

**Insert this task between Task 17 and Task 18.** Apply BEFORE Task 18's deletion step.

**Files:**
- Modify: `src/hydra_suite/core/identity/pose/features.py`
- Create: `tests/test_pose_features_with_new_cache.py`

- [ ] **Step 1: Identify the call sites in `features.py`**

```bash
grep -n "pose_props_cache\|get_frame\|pose_keypoints\|detection_ids" \
    src/hydra_suite/core/identity/pose/features.py
```

Expected: at least one call site in `build_pose_detection_keypoint_map`.

- [ ] **Step 2: Replace the cache read with a `PoseCache` adapter**

Add this adapter at the top of `features.py`:

```python
def _frame_dict_from_pose_cache(
    pose_cache,
    detection_cache,
    frame_idx: int,
) -> dict | None:
    """Adapter: return a frame dict shaped like the legacy IndividualPropertiesCache.

    Reads PoseResult from the new PoseCache and the matching detection_ids array
    from DetectionCache. Returns None if either cache has no entry for this frame.
    """
    pose_result = pose_cache.read_frame(frame_idx)
    obb_result = detection_cache.read_frame(frame_idx)
    if pose_result is None or obb_result is None:
        return None
    return {
        "detection_ids": obb_result.detection_ids,
        "pose_keypoints": pose_result.keypoints,   # (D, K, 3)
        "pose_valid_mask": pose_result.valid_mask, # (D,)
    }
```

Update `build_pose_detection_keypoint_map` to take `pose_cache` and `detection_cache` (both new) and call `_frame_dict_from_pose_cache` instead of the deleted `pose_props_cache.get_frame`.

- [ ] **Step 3: Update all callers**

Search for callers:
```bash
grep -rn "build_pose_detection_keypoint_map" src/ tests/
```

The only caller is `core/tracking/worker.py`. Update its call sites to pass the new caches (held by `InferenceRunner._caches.pose` and `_caches.detection`).

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_pose_features_with_new_cache.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/identity/pose/features.py tests/test_pose_features_with_new_cache.py
git commit -m "refactor(pose): rewire features.py to read from new PoseCache + DetectionCache"
```

---

### Correction 19 — New Task 17c: Rewire `core/identity/properties/export.py`

`export.py` (kept as-is) imports `CNNIdentityCache` and `DetectedPropertiesCache` from deleted modules. The CSV-export pipeline that produces wide-format identity columns will crash at import.

**Files:**
- Modify: `src/hydra_suite/core/identity/properties/export.py`
- Create: `tests/test_properties_export_with_new_caches.py`

- [ ] **Step 1: Identify the imports**

```bash
grep -n "from hydra_suite.core.identity.classification.cnn\|from .detected_cache\|from .cache" \
    src/hydra_suite/core/identity/properties/export.py
```

- [ ] **Step 2: Replace with new-cache imports**

```python
# Old:
# from hydra_suite.core.identity.classification.cnn import CNNIdentityCache
# from .detected_cache import DetectedPropertiesCache
# from .cache import IndividualPropertiesCache

# New:
from hydra_suite.core.inference.cache.cnn import CNNCache
from hydra_suite.core.inference.cache.pose import PoseCache
from hydra_suite.core.inference.cache.detection import DetectionCache
```

The export functions read per-frame predictions and emit CSV columns. The new `CNNCache.read_frame()` returns `list[CNNDetectionPrediction]` (Phase 1 type). Update the column-generation loop to traverse `predictions[i].factors[j].raw_probabilities` and `class_names`. Calibration must be applied in the export step using the same `IdentityEvidenceBuilder` logic — extract the calibration helper into a shared util to avoid duplication.

- [ ] **Step 3: Run tests + commit**

---

### Correction 20 — New Task 17d: Rewire `core/identity/dataset/oriented_video.py`

The training-dataset generator imports `DetectionCache` from `hydra_suite.data.detection_cache` (deleted) to read OBB crops for dataset building.

**Files:**
- Modify: `src/hydra_suite/core/identity/dataset/oriented_video.py`

- [ ] **Step 1: Replace the import**

```python
# Old:
# from ....data.detection_cache import DetectionCache

# New:
from hydra_suite.core.inference.cache.detection import DetectionCache
```

- [ ] **Step 2: Update field-access call sites**

The legacy `DetectionCache.get_frame()` returned a 12-tuple (see audit findings). The new `DetectionCache.read_frame()` returns an `OBBResult`. Update the dataset generator to access `obb.centroids`, `obb.corners`, `obb.detection_ids` instead of tuple unpacking.

- [ ] **Step 3: Run dataset-generation smoke test on a small fixture; commit.**

---

### Correction 21 — New Task 17e: Rewire `core/tracking/optimizer.py` and `optimizer_workers.py`

Both files import `DetectionFilter` from deleted `core/detectors/detection_filter.py` and `DetectionCache` from deleted `data/detection_cache.py`.

**Files:**
- Modify: `src/hydra_suite/core/tracking/optimizer.py`
- Modify: `src/hydra_suite/core/tracking/optimizer_workers.py`
- Create: `src/hydra_suite/core/inference/api.py` (public helpers used outside `core/inference/`)

- [ ] **Step 1: Expose `apply_detection_filter` shim**

Add to `src/hydra_suite/core/inference/api.py`:

```python
"""Public helpers for callers outside core/inference/.

Keep this surface minimal: each helper exists to support a specific kept consumer
that cannot directly depend on the internal stages module.
"""
from __future__ import annotations

from .config import OBBConfig
from .result import OBBResult
from .stages.filtering import filter_detections


def apply_detection_filter(raw: OBBResult, config: OBBConfig) -> OBBResult:
    """Filter raw OBB detections using the same logic the runner uses internally.

    Used by core/tracking/optimizer.py to score parameter configurations against
    cached detections. Pure function — no I/O, no model loading.
    """
    return filter_detections(raw, config, roi_mask=None)
```

- [ ] **Step 2: Update optimizer imports**

```python
# Old:
# from hydra_suite.core.detectors import DetectionFilter
# from hydra_suite.data.detection_cache import DetectionCache

# New:
from hydra_suite.core.inference.api import apply_detection_filter
from hydra_suite.core.inference.cache.detection import DetectionCache
```

- [ ] **Step 3: Update method bodies**

Find every `DetectionFilter(...).filter(raw)` call and replace with `apply_detection_filter(raw, config)`. The `DetectionCache.get_frame()` calls switch to `read_frame()` and access `OBBResult` fields directly.

- [ ] **Step 4: Run optimizer tests; commit.**

---

### Correction 22 — New Task 17f: Add public helper for `posekit/gui/workers.py`

`posekit/gui/workers.py` does a lazy import of `build_runtime_config` and `create_pose_backend_from_config` from deleted `core/identity/pose/api.py` to do single-image pose prediction in PoseKit's labeling UI. The spec only addresses TrackerKit's `worker.py`; PoseKit is otherwise out of scope.

**Files:**
- Modify: `src/hydra_suite/core/inference/api.py` (add helper)
- Modify: `src/hydra_suite/posekit/gui/workers.py`

- [ ] **Step 1: Add `predict_pose_for_image` helper**

```python
# In src/hydra_suite/core/inference/api.py, append:

from .config import PoseConfig
from .runtime import RuntimeContext


def predict_pose_for_image(image, pose_config: PoseConfig) -> "PoseResult":
    """One-shot pose prediction on a single image, used by PoseKit labeling UI.

    Loads a pose model, runs inference once, and discards the model. NOT for
    batch use — call `InferenceRunner.run_realtime` if you need persistent state.
    """
    from .runner import _load_pose_model
    from .stages.pose import run_pose
    runtime = RuntimeContext.from_config_fields(
        compute_runtime=pose_config.yolo.compute_runtime if pose_config.backend == "yolo"
        else pose_config.sleap.compute_runtime
    )
    model = _load_pose_model(pose_config, runtime)
    try:
        # Single-frame, single-detection: synthetic OBB covering full image.
        return run_pose([image], None, model, pose_config, runtime)[0]
    finally:
        del model
```

- [ ] **Step 2: Update PoseKit workers**

```python
# In posekit/gui/workers.py, replace the lazy import:
# Old:
# from hydra_suite.core.identity.pose.api import build_runtime_config, create_pose_backend_from_config

# New:
from hydra_suite.core.inference.api import predict_pose_for_image
```

Update the call sites accordingly. The `try/except ImportError` block around the old import can be kept for one cycle as a safety net.

- [ ] **Step 3: Smoke-test PoseKit on a real image; commit.**

---

### Correction 23 — New Task 17g: Construct `StreamingAnalysisPayload` inside `runner.run_realtime()`

`tracking/streaming_payload.py` is "kept as-is" but its only constructor (`build_streaming_payload` in deleted `live_features.py`) goes away. The GUI streaming overlay (visualization in TrackerKit) reads `StreamingAnalysisPayload`.

**Files:**
- Modify: `src/hydra_suite/core/inference/runner.py`
- Modify: `src/hydra_suite/core/inference/result.py` (add optional payload field to `FrameResult`)

- [ ] **Step 1: Add `streaming_payload: StreamingAnalysisPayload | None = None` to `FrameResult`**

Default `None` so non-RT mode doesn't pay the construction cost.

- [ ] **Step 2: In `runner.run_realtime()`, construct the payload before returning**

```python
def run_realtime(self, frame) -> FrameResult:
    # ... existing inference code ...
    result = _build_frame_result(...)
    # Construct streaming payload for GUI overlay
    from hydra_suite.core.tracking.streaming_payload import StreamingAnalysisPayload
    result.streaming_payload = StreamingAnalysisPayload.from_frame_result(result, frame)
    return result
```

- [ ] **Step 3: Add `StreamingAnalysisPayload.from_frame_result(...)` classmethod**

In `core/tracking/streaming_payload.py`, add a constructor that derives the payload from a `FrameResult` + raw frame. This is a pure transformation — no inference, no I/O. Keeps the payload class itself "unchanged" in spirit (constructor logic is added, not modified).

- [ ] **Step 4: Verify GUI overlay still renders. Commit.**

---

### Correction 24 — Task 17/18: Backward pass is a separate worker, NOT a method call

The original plan's worker.py pseudocode shows `self._run_backward_pass()` and `self._run_consensus_resolution()` calls at the end of `run()`. These methods do not exist on `TrackingWorker`. The actual mechanism:

1. The GUI (`trackerkit/gui/main_window.py`) launches a forward `TrackingWorker(backward_mode=False)`.
2. On `finished` signal, the GUI launches a second `TrackingWorker(backward_mode=True)` reusing the same `cache_dir`.
3. The backward worker's `run()` is the same flow as the forward worker's, but with `caches_all_valid()` asserted (no inference re-run) and frame iteration reversed.
4. After both finish, the GUI runs `core/post/identity_postprocess.consensus_resolve(forward_csv, backward_csv)` in-process. This is unchanged — operates on DataFrames, not on inference state.

**Modify Task 17 Step 5** — remove the `self._run_backward_pass()` line. The forward worker ends after `runner.close()`. The GUI orchestration is documented but does not change in this refactor.

**Add to Task 17 Step 5** — make `_run_with_new_pipeline` respect `self.backward_mode`:

```python
    def _run_with_new_pipeline(self, ...) -> None:
        config = InferenceConfig.from_json(config_path)
        runner = InferenceRunner(config, cache_dir=cache_dir)

        try:
            if self.backward_mode:
                # Backward pass MUST find pre-computed caches. Refuse to run inference.
                if not runner.caches_all_valid():
                    raise RuntimeError(
                        "Backward pass requires valid forward-pass caches. "
                        "Run forward pass first."
                    )
                frame_iter = reversed(range(total_frames))
            else:
                if not runner.caches_all_valid():
                    runner.run_batch_pass(video_path, progress_cb=...)
                frame_iter = range(total_frames)

            for frame_idx in frame_iter:
                frame_result = runner.load_frame(frame_idx)
                # ... rest unchanged
        finally:
            runner.close()
```

**Add a test** to `tests/test_worker_inference_integration.py`:

```python
def test_backward_mode_refuses_to_run_inference(tmp_path):
    """Backward pass must NOT call run_batch_pass — caches must already exist."""
    from hydra_suite.core.tracking.worker import TrackingWorker

    mock_runner = MagicMock()
    mock_runner.caches_all_valid.return_value = False  # caches missing

    with patch("hydra_suite.core.tracking.worker.InferenceRunner", return_value=mock_runner):
        with patch("hydra_suite.core.tracking.worker.InferenceConfig") as mock_cfg_cls:
            mock_cfg_cls.from_json.return_value = MagicMock(realtime=False, cnn=[])
            worker_obj = TrackingWorker.__new__(TrackingWorker)
            worker_obj._identity_builders = []
            worker_obj.backward_mode = True
            with pytest.raises(RuntimeError, match="forward-pass caches"):
                worker_obj._run_with_new_pipeline(
                    video_path=tmp_path / "video.mp4",
                    config_path=str(tmp_path / "cfg.json"),
                    cache_dir=tmp_path,
                    total_frames=2,
                )

    mock_runner.run_batch_pass.assert_not_called()
```

---

### Correction 25 — New Task 18a: Update `__init__.py` re-exports BEFORE deletion

This step is mandatory and goes BEFORE Task 18's `git rm` step. Without it, `python -c "import hydra_suite"` crashes the moment a deleted file is removed because the surviving `__init__.py` files still try to re-export deleted symbols.

**Files (all modified, none created):**
- `src/hydra_suite/core/detectors/__init__.py`
- `src/hydra_suite/core/identity/classification/__init__.py`
- `src/hydra_suite/core/identity/pose/__init__.py`
- `src/hydra_suite/core/identity/properties/__init__.py`
- `src/hydra_suite/data/__init__.py`

- [ ] **Step 1: `core/detectors/__init__.py`**

Drop these re-exports:
```python
# DELETE these lines:
# from .detection_filter import DetectionFilter
# from .factory import create_detector
# from .yolo_detector import YOLOOBBDetector
```

Keep:
```python
from .bg_detector import ObjectDetector
from .bg_optimizer import optimize_background
```

If a downstream consumer needs `DetectionFilter`, point them to `hydra_suite.core.inference.api.apply_detection_filter` (added in Correction 21).

- [ ] **Step 2: `core/identity/classification/__init__.py`**

Drop `CNNIdentityBackend`, `CNNIdentityCache`, `CNNIdentityConfig`, `ClassPrediction`, `TrackCNNHistory`, `apply_cnn_identity_cost`, `HeadTailAnalyzer`. Keep `ClassifierBackend` (from `backend.py`), `apriltag` exports, `errors`.

- [ ] **Step 3: `core/identity/pose/__init__.py`**

Drop `YoloNativeBackend`, `SleapServiceBackend`, `auto_export_yolo_model`, `auto_export_sleap_model`, `build_runtime_config`, `create_pose_backend_from_config`. Keep `quality`, `artifacts`, `types`.

- [ ] **Step 4: `core/identity/properties/__init__.py`**

Drop `IndividualPropertiesCache`, `DetectedPropertiesCache`. The new pose cache is internal to `core/inference/cache/pose.py` — do not re-export.

- [ ] **Step 5: `data/__init__.py`**

Drop `DetectionCache`, `TagObservationCache`. If a public alias is needed for one cycle:

```python
# Backwards-compat shim — REMOVE after one release cycle.
from hydra_suite.core.inference.cache.detection import DetectionCache as DetectionCache  # noqa: F401
from hydra_suite.core.inference.cache.apriltag import AprilTagCache as TagObservationCache  # noqa: F401
```

- [ ] **Step 6: Run import smoke test**

```bash
python -c "import hydra_suite; print('ok')"
python -c "from hydra_suite.core import detectors, identity; print('ok')"
python -c "import trackerkit, posekit, classkit, refinekit, detectkit, filterkit"
```

Expected: no `ImportError` from any entry point. Run the full test suite to confirm.

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/core/detectors/__init__.py \
        src/hydra_suite/core/identity/classification/__init__.py \
        src/hydra_suite/core/identity/pose/__init__.py \
        src/hydra_suite/core/identity/properties/__init__.py \
        src/hydra_suite/data/__init__.py
git commit -m "refactor(inference): drop deleted symbols from __init__.py re-exports"
```

---

### Correction 26 — Task 18: Test migration list

The original Task 18 is silent on tests. At least five test files import deleted modules and will crash. Add this step to Task 18 between current Step 3 (`git rm` legacy files) and current Step 4 (fix broken imports):

- [ ] **Step 3.5: Migrate or delete affected tests**

Test files importing deleted modules:

| Test file | Imports from deleted | Action |
|---|---|---|
| `tests/test_detection_cache.py` | `data.detection_cache` | Rewrite to test `core/inference/cache/detection.py` instead — most assertions still apply (npz round-trip, key invalidation). Move to `tests/test_inference_cache_detection.py`. |
| `tests/test_tag_observation_cache.py` | `data.tag_observation_cache` | Same: rewrite against `core/inference/cache/apriltag.py`. |
| `tests/test_individual_properties_cache.py` | `core/identity/properties/cache.py` | Rewrite against `core/inference/cache/pose.py`. |
| `tests/test_pose_pipeline.py` | `core/tracking/pose_pipeline.py` (deleted) | Delete. The pose pipeline is replaced by `runner.py` IndividualWorker; coverage moves to `tests/test_inference_runner_batch.py`. |
| `tests/test_tag_features.py` | `core/tracking/tag_features.py` (deleted) | Delete. Tag-evidence logic moved to `tests/test_identity_evidence_builder.py`. |

- [ ] **Step 3.6: Search for any straggler test imports**

```bash
grep -rn "from hydra_suite.core.tracking.\(detection_phase\|precompute\|pose_pipeline\|live_features\|cnn_features\|tag_features\|evidence_emitter\)\|from hydra_suite.core.detectors.\(yolo_detector\|factory\|detection_filter\)\|from hydra_suite.core.identity.classification.\(cnn\|headtail\)\|from hydra_suite.core.identity.pose.\(api\|backends\)\|from hydra_suite.data.\(detection_cache\|tag_observation_cache\)\|from hydra_suite.core.identity.properties.\(cache\|detected_cache\)" tests/
```

Expected: no matches. Any straggler should be migrated or deleted.

- [ ] **Step 3.7: Commit the test migrations**

```bash
git add tests/
git commit -m "test(inference): migrate cache tests to new module; delete obsolete pipeline tests"
```

---

### Correction 27 — Task 18: Final import-graph verification before deletion

Even with Corrections 18–26 applied, a missed importer can still crash the build. Add this verification step BEFORE the `git rm` of legacy files:

- [ ] **Step 2.5: Dry-run the import graph**

```bash
# 1. Run the full test suite with feature flag ON
USE_NEW_INFERENCE_PIPELINE=1 python -m pytest tests/ -v -m "not benchmark"

# 2. Smoke-import every public entry point
for entry in hydra_suite trackerkit posekit classkit refinekit detectkit filterkit; do
    python -c "import $entry" || echo "FAIL: $entry"
done

# 3. Smoke-import every kept module that previously imported a soon-to-be-deleted file
for mod in \
    hydra_suite.core.tracking.optimizer \
    hydra_suite.core.tracking.optimizer_workers \
    hydra_suite.core.identity.properties.export \
    hydra_suite.core.identity.dataset.oriented_video \
    hydra_suite.core.identity.pose.features \
    hydra_suite.posekit.gui.workers \
    hydra_suite.core.tracking.streaming_payload; do
    python -c "import $mod" || echo "FAIL: $mod"
done

# 4. Final scan for any remaining importers of soon-to-be-deleted modules
grep -rn "from hydra_suite.core.tracking.\(detection_phase\|precompute\|pose_pipeline\|live_features\|cnn_features\|tag_features\|evidence_emitter\)" src/
grep -rn "from hydra_suite.core.detectors.\(yolo_detector\|factory\|detection_filter\)" src/
grep -rn "from hydra_suite.core.identity.classification.\(cnn\|headtail\)" src/
grep -rn "from hydra_suite.core.identity.pose.\(api\|backends\)" src/
grep -rn "from hydra_suite.data.\(detection_cache\|tag_observation_cache\)" src/
grep -rn "from hydra_suite.core.identity.properties.\(cache\|detected_cache\)" src/
```

Expected: every command exits cleanly. Any `FAIL: …` output or non-empty `grep` result must be resolved before proceeding to the `git rm` step.

---

*The 14 downstream-compatibility issues identified in the 2026-05-04 audit are now documented as Corrections 14–27. Apply them in order. Tasks 17b–17g and 18a are new; the rest amend existing tasks. After applying every correction in this section AND the prior "Type Consistency" and "Bug and Performance" sections, the plan is safe to execute end-to-end.*
