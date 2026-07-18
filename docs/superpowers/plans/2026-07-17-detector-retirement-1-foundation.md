# Detector Retirement — Plan 1: Foundation (Phases A + B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `hydra_suite.core.inference` fully self-contained (stop importing `core/detectors`) and add the shared config + batched-detect entrypoints that later plans migrate consumers onto — without changing any pipeline behavior.

**Architecture:** Relocate the self-contained `_direct_obb_runtime` module into `core/inference` and repoint the one internal import (Phase A). Then promote the params→`InferenceConfig` builder off `TrackingWorker` into `config.py`, add an OBB-only config helper, fix the never-run `predict_pose_for_image` one-shot pose helper, and add a batched-in-memory `detect_batch` method to `InferenceRunner` that mirrors `run_realtime`'s detect+filter prefix (Phase B).

**Tech Stack:** Python 3, PySide6 (unaffected here), ultralytics/torch/numpy, pytest.

## Global Constraints

- No behavior change to the tracking/inference pipeline. `run_realtime` / `run_batch_pass` outputs must remain byte-identical; these tasks only add or relocate code.
- Do not import from any `core/detectors` module in `core/inference` after Task 1.
- The pytest monkeypatch target `runtime_artifacts._create_direct_executor` must remain valid (see `tests/test_inference_obb_artifacts.py:83,237`).
- Follow existing project conventions; run `make format` before each commit. Tests via `python -m pytest ... -q` (benchmarks excluded by default).
- This plan runs on `main` after `plans/2026-07-16-bgsub-inference-stage.md` and `plans/2026-07-16-legacy-batching-vestige-removal.md` have landed (they have).

---

## File Structure

- `src/hydra_suite/core/inference/direct_executors.py` — **new** (relocated from `core/detectors/_direct_obb_runtime.py` via `git mv`). Direct ONNX/TRT OBB + detect executors. Self-contained (stdlib + numpy + torch only).
- `src/hydra_suite/core/inference/runtime_artifacts.py` — **modify** the `_create_direct_executor` import (line 241) + the module docstring (lines 1–44).
- `src/hydra_suite/core/inference/config.py` — **add** public `build_inference_config_from_params(params)` and `build_obb_only_config(...)`.
- `src/hydra_suite/core/tracking/worker.py` — **remove** the `_build_inference_config_from_params` method (lines 4364–4665); update its one call site (line 984) to the new module function.
- `src/hydra_suite/core/inference/api.py` — **fix** `predict_pose_for_image` (lines 41–113).
- `src/hydra_suite/core/inference/runner.py` — **add** `InferenceRunner.detect_batch(...)`.
- `tools/diag_trt_vs_cuda.py`, `tools/diag_trt_vs_cuda_full.py`, `tools/compare_runtimes.py` — **repoint** `Direct*Executor` imports.
- Tests: `tests/test_inference_direct_executors_location.py` (new), `tests/test_inference_config_from_params.py` (new), `tests/test_inference_api_pose.py` (new), `tests/test_inference_runner_detect_batch.py` (new). Existing `tests/test_inference_obb_artifacts.py` must still pass unchanged.

---

## Phase A — Port the runtime layer out of `core/detectors`

### Task 1: Relocate `_direct_obb_runtime.py` into `core/inference`

The module is self-contained: its only imports are stdlib (`ast`, `json`, `struct`, `pathlib`, `typing`), `numpy`, and `torch` (under `TYPE_CHECKING`). It imports nothing from `core/detectors`. So the relocation is a pure move + one import repoint.

**Files:**
- Move: `src/hydra_suite/core/detectors/_direct_obb_runtime.py` → `src/hydra_suite/core/inference/direct_executors.py`
- Modify: `src/hydra_suite/core/inference/runtime_artifacts.py:241-243`
- Modify: `tools/diag_trt_vs_cuda.py`, `tools/diag_trt_vs_cuda_full.py`, `tools/compare_runtimes.py`
- Test: `tests/test_inference_direct_executors_location.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `hydra_suite.core.inference.direct_executors` exporting `create_direct_obb_executor(*, runtime, artifact_path, imgsz, class_names=None, class_count=None, pt_model=None)`, `create_direct_detect_executor(*, runtime, artifact_path, imgsz, class_names=None, class_count=None)`, and classes `DirectONNXOBBExecutor`, `DirectTensorRTOBBExecutor`, `DirectPyTorchCUDAOBBExecutor`, `DirectONNXDetectExecutor`, `DirectTensorRTDetectExecutor`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_direct_executors_location.py
"""Guards the Phase A relocation: direct executors live in core/inference,
and core/inference no longer imports from core/detectors."""
import ast
from pathlib import Path

import hydra_suite.core.inference.direct_executors as de


def test_direct_executor_factories_importable_from_inference():
    assert hasattr(de, "create_direct_obb_executor")
    assert hasattr(de, "create_direct_detect_executor")
    for name in (
        "DirectONNXOBBExecutor",
        "DirectTensorRTOBBExecutor",
        "DirectPyTorchCUDAOBBExecutor",
        "DirectONNXDetectExecutor",
        "DirectTensorRTDetectExecutor",
    ):
        assert hasattr(de, name), name


def test_inference_package_does_not_import_core_detectors():
    inference_dir = Path(de.__file__).parent
    offenders = []
    for py in inference_dir.rglob("*.py"):
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.ImportFrom):
                mod = node.module
            elif isinstance(node, ast.Import):
                mod = ",".join(a.name for a in node.names)
            if mod and "core.detectors" in mod:
                offenders.append(f"{py.name}:{node.lineno} -> {mod}")
    assert not offenders, "core/inference must not import core/detectors: " + "; ".join(offenders)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_direct_executors_location.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.core.inference.direct_executors'`.

- [ ] **Step 3: Move the module**

Run:
```bash
git mv src/hydra_suite/core/detectors/_direct_obb_runtime.py \
       src/hydra_suite/core/inference/direct_executors.py
```

- [ ] **Step 4: Repoint the import in `runtime_artifacts.py`**

In `src/hydra_suite/core/inference/runtime_artifacts.py`, replace the import inside `_create_direct_executor` (lines 241–244):

```python
    from hydra_suite.core.detectors._direct_obb_runtime import (
        create_direct_detect_executor,
        create_direct_obb_executor,
    )
```

with:

```python
    from .direct_executors import (
        create_direct_detect_executor,
        create_direct_obb_executor,
    )
```

- [ ] **Step 5: Repoint the three tools scripts**

In each of `tools/diag_trt_vs_cuda.py`, `tools/diag_trt_vs_cuda_full.py`, `tools/compare_runtimes.py`, change the import of the `Direct*` executor classes from
`from hydra_suite.core.detectors._direct_obb_runtime import (...)` to
`from hydra_suite.core.inference.direct_executors import (...)` (keep the imported names identical).

- [ ] **Step 6: Run the relocation test + the existing artifact test**

Run:
```bash
python -m pytest tests/test_inference_direct_executors_location.py tests/test_inference_obb_artifacts.py -q
```
Expected: PASS. (`test_inference_obb_artifacts.py` patches `runtime_artifacts._create_direct_executor`, a name untouched by this move, so it still passes.)

- [ ] **Step 7: Fix the module docstring in `runtime_artifacts.py`**

The current docstring (lines 1–44) says both that the module "deliberately does NOT import from `core/detectors`" (line 9) and that the executors are "ported in `core/detectors/_direct_obb_runtime.py`" (lines 6, 34, 42). Both `core/detectors` references are now false. Update the docstring so it states the executors live in `core/inference/direct_executors.py` and that this module delegates to them (replace the three `core/detectors/_direct_obb_runtime.py` mentions with `core/inference/direct_executors.py`, and change line 9's "does NOT import from `core/detectors`" to "delegates to the sibling `direct_executors` module").

- [ ] **Step 8: Verify no stale references remain**

Run:
```bash
grep -rn "_direct_obb_runtime\|detectors._direct_obb" src/ tools/
```
Expected: no output. (`tests/test_detectors_engine.py` may still reference it by file path — that is Plan 4's Phase F concern; do not touch tests here.)

- [ ] **Step 9: Commit**

```bash
make format
git add -A
git commit -m "refactor(inference): relocate direct OBB executors into core/inference"
```

---

## Phase B — Shared config + one-shot pose fix + batched detect

### Task 2: Promote the params→InferenceConfig builder into `config.py`

`TrackingWorker._build_inference_config_from_params` (worker.py:4364–4665) calls no `self.*` helpers — it is a pure function of `params`. Move it verbatim into `config.py` as a public module function; the dataclasses and `migrate_runtime_to_tier` it uses already live in `config.py`, so the local `from hydra_suite.core.inference.config import ...` lines become unnecessary.

**Files:**
- Modify: `src/hydra_suite/core/inference/config.py` (add function; ensure `import os`)
- Modify: `src/hydra_suite/core/tracking/worker.py` (remove method lines 4364–4665; update call site line 984)
- Test: `tests/test_inference_config_from_params.py`

**Interfaces:**
- Consumes: dataclasses in `config.py`, `migrate_runtime_to_tier`.
- Produces: `build_inference_config_from_params(params: dict) -> InferenceConfig` (module-level in `config.py`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_config_from_params.py
from hydra_suite.core.inference.config import (
    InferenceConfig,
    build_inference_config_from_params,
)


def test_direct_obb_minimal_params():
    cfg = build_inference_config_from_params(
        {
            "DETECTION_METHOD": "yolo_obb",
            "YOLO_OBB_MODE": "direct",
            "YOLO_OBB_DIRECT_MODEL_PATH": "some.pt",
            "COMPUTE_RUNTIME": "cpu",
            "MAX_TARGETS": 8,
        }
    )
    assert isinstance(cfg, InferenceConfig)
    assert cfg.obb is not None and cfg.obb.mode == "direct"
    assert cfg.obb.direct.model_path == "some.pt"
    # No headtail/cnn/pose enabled -> those stay unset (OBB-only by omission).
    assert cfg.headtail is None
    assert cfg.cnn_phases == []
    assert cfg.pose is None
    # raw cap = 2*MAX_TARGETS, final cap = MAX_TARGETS (legacy parity).
    assert cfg.obb.max_detections == 8
    assert cfg.obb.raw_detection_cap == 16
    assert cfg.runtime_tier == "cpu"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_config_from_params.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_inference_config_from_params'`.

- [ ] **Step 3: Add the function to `config.py`**

Ensure `import os` is present at the top of `config.py` (add it if absent). Then add, at module level (e.g. directly after `_dict_to_config`), the function below. It is the verbatim body of `worker._build_inference_config_from_params` with `self` removed and the redundant same-module import lines deleted:

```python
def build_inference_config_from_params(params: dict) -> InferenceConfig:
    """Build an InferenceConfig from a tracking-worker params dict.

    Maps legacy YOLO/headtail/CNN/pose/AprilTag params to the structured
    InferenceConfig dataclasses consumed by InferenceRunner. Stages whose
    params are absent/disabled stay unset, so an OBB-only params dict yields
    an OBB-only config (headtail=None, cnn_phases=[], pose=None).
    """
    compute_runtime = str(params.get("COMPUTE_RUNTIME", "cpu"))
    _raw_tier = str(params.get("RUNTIME_TIER", "") or "").strip().lower()
    runtime_tier = (
        _raw_tier
        if _raw_tier in {"cpu", "gpu", "gpu_fast"}
        else migrate_runtime_to_tier({compute_runtime})
    )
    obb_mode = str(params.get("YOLO_OBB_MODE", "direct")).strip().lower()
    if obb_mode not in {"direct", "sequential"}:
        obb_mode = "direct"

    direct_model_path = str(
        params.get(
            "YOLO_OBB_DIRECT_MODEL_PATH",
            params.get("YOLO_MODEL_PATH", "yolo26s-obb.pt"),
        )
        or "yolo26s-obb.pt"
    )
    yolo_conf = float(params.get("YOLO_CONFIDENCE_THRESHOLD", 0.25))
    yolo_iou = float(params.get("YOLO_IOU_THRESHOLD", 0.7))
    min_obj = float(params.get("MIN_OBJECT_SIZE", 0.0))
    max_obj = float(params.get("MAX_OBJECT_SIZE", float("inf")) or float("inf"))
    max_targets = max(1, int(params.get("MAX_TARGETS", 8)))
    raw_cap = 2 * max_targets
    max_dets = max_targets

    _target_classes_raw = params.get("YOLO_TARGET_CLASSES", None)
    target_classes = (
        [int(c) for c in _target_classes_raw] if _target_classes_raw else []
    )

    _adv = params.get("ADVANCED_CONFIG", {}) or {}
    if _adv.get("enable_aspect_ratio_filtering", False):
        ref_ar = float(_adv.get("reference_aspect_ratio", 2.0))
        min_ar = ref_ar * float(_adv.get("min_aspect_ratio_multiplier", 0.5))
        max_ar = ref_ar * float(_adv.get("max_aspect_ratio_multiplier", 2.0))
    else:
        min_ar, max_ar = 0.0, float("inf")

    if obb_mode == "sequential":
        detect_path = str(params.get("YOLO_DETECT_MODEL_PATH", "") or "")
        crop_path = str(
            params.get("YOLO_CROP_OBB_MODEL_PATH", "") or direct_model_path
        )
        obb_cfg = OBBConfig(
            mode="sequential",
            sequential=OBBSequentialConfig(
                detect_model_path=detect_path,
                obb_model_path=crop_path,
                detect_compute_runtime=compute_runtime,
                obb_compute_runtime=compute_runtime,
                detect_confidence_threshold=float(
                    params.get("YOLO_SEQ_DETECT_CONF_THRESHOLD", 0.25)
                ),
                obb_confidence_threshold=yolo_conf,
                detect_image_size=int(params.get("YOLO_SEQ_DETECT_IMGSZ", 0)),
                crop_pad_ratio=float(params.get("YOLO_SEQ_CROP_PAD_RATIO", 0.15)),
                min_crop_size_px=float(params.get("YOLO_SEQ_MIN_CROP_SIZE_PX", 64.0)),
                enforce_square_crop=bool(
                    params.get("YOLO_SEQ_ENFORCE_SQUARE_CROP", True)
                ),
                stage2_image_size=int(params.get("YOLO_SEQ_STAGE2_IMGSZ", 160)),
                stage2_batch_size=(
                    int(params["YOLO_SEQ_INDIVIDUAL_BATCH_SIZE"])
                    if params.get("YOLO_SEQ_INDIVIDUAL_BATCH_SIZE")
                    else None
                ),
            ),
            target_classes=target_classes,
            confidence_threshold=yolo_conf,
            iou_threshold=yolo_iou,
            min_object_size=min_obj,
            max_object_size=max_obj,
            min_aspect_ratio=min_ar,
            max_aspect_ratio=max_ar,
            max_detections=max_dets,
            raw_detection_cap=raw_cap,
        )
    else:
        obb_cfg = OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(
                model_path=direct_model_path,
                compute_runtime=compute_runtime,
                confidence_floor=1e-3,
                confidence_threshold=yolo_conf,
            ),
            target_classes=target_classes,
            confidence_threshold=yolo_conf,
            iou_threshold=yolo_iou,
            min_object_size=min_obj,
            max_object_size=max_obj,
            min_aspect_ratio=min_ar,
            max_aspect_ratio=max_ar,
            max_detections=max_dets,
            raw_detection_cap=raw_cap,
        )

    headtail_model_path = str(params.get("YOLO_HEADTAIL_MODEL_PATH", "") or "").strip()
    headtail_cfg = None
    if headtail_model_path and os.path.exists(headtail_model_path):
        ht_runtime = str(
            params.get("HEADTAIL_COMPUTE_RUNTIME", params.get("COMPUTE_RUNTIME", "cpu"))
        )
        headtail_cfg = HeadTailConfig(
            model_path=headtail_model_path,
            compute_runtime=ht_runtime,
            confidence_threshold=float(params.get("YOLO_HEADTAIL_CONF_THRESHOLD", 0.5)),
            candidate_confidence_threshold=float(
                params.get(
                    "YOLO_HEADTAIL_DETECT_CONF_THRESHOLD",
                    params.get("YOLO_CONFIDENCE_THRESHOLD", 0.25),
                )
            ),
            batch_size=int(params.get("HEADTAIL_BATCH_SIZE", 64)),
            canonical_aspect_ratio=float(
                params.get("ADVANCED_CONFIG", {}).get("reference_aspect_ratio", 2.0)
            ),
            canonical_margin=float(
                params.get("ADVANCED_CONFIG", {}).get(
                    "yolo_headtail_canonical_margin", 1.3
                )
            ),
        )

    cnn_phases: list[CNNConfig] = []
    cnn_runtime = str(
        params.get("CNN_COMPUTE_RUNTIME", params.get("COMPUTE_RUNTIME", "cpu"))
    )
    for cnn_cfg_dict in params.get("CNN_CLASSIFIERS", []):
        cnn_model_path = str(cnn_cfg_dict.get("model_path", "")).strip()
        if not cnn_model_path or not os.path.exists(cnn_model_path):
            continue
        cnn_label = str(cnn_cfg_dict.get("label", "cnn_identity"))
        cnn_phases.append(
            CNNConfig(
                label=cnn_label,
                model_path=cnn_model_path,
                compute_runtime=cnn_runtime,
                confidence_threshold=float(cnn_cfg_dict.get("confidence", 0.5)),
                batch_size=int(cnn_cfg_dict.get("batch_size", 64)),
                scoring_mode=str(cnn_cfg_dict.get("scoring_mode", "atomic")),
                match_bonus=float(cnn_cfg_dict.get("match_bonus", 0.1)),
                mismatch_penalty=float(cnn_cfg_dict.get("mismatch_penalty", 0.3)),
                calibration_temperature=float(
                    cnn_cfg_dict.get(
                        "calibration_temperature", cnn_cfg_dict.get("temperature", 1.0)
                    )
                ),
            )
        )

    pose_cfg = None
    if bool(params.get("ENABLE_POSE_EXTRACTOR", False)):
        pose_model_type = str(params.get("POSE_MODEL_TYPE", "")).strip().lower()
        pose_runtime = str(
            params.get("POSE_COMPUTE_RUNTIME", params.get("COMPUTE_RUNTIME", "cpu"))
        )
        common_pose_kwargs = dict(
            skeleton_file=str(params.get("POSE_SKELETON_FILE", "") or "").strip(),
            crop_padding=float(params.get("INDIVIDUAL_CROP_PADDING", 0.1)),
            suppress_foreign_regions=bool(
                params.get("SUPPRESS_FOREIGN_OBB_REGIONS", True)
            ),
            min_keypoint_confidence=float(params.get("POSE_MIN_KPT_CONF_VALID", 0.2)),
            min_valid_keypoints=int(params.get("POSE_DIRECTION_MIN_VALID_KEYPOINTS", 1)),
            anterior_keypoints=list(
                params.get("POSE_DIRECTION_ANTERIOR_KEYPOINTS", []) or []
            ),
            posterior_keypoints=list(
                params.get("POSE_DIRECTION_POSTERIOR_KEYPOINTS", []) or []
            ),
            ignore_keypoints=list(params.get("POSE_IGNORE_KEYPOINTS", []) or []),
            overrides_headtail=bool(params.get("POSE_OVERRIDES_HEADTAIL", True)),
        )
        sleap_model_path = str(
            params.get("POSE_SLEAP_MODEL_DIR", params.get("POSE_MODEL_DIR", "")) or ""
        ).strip()
        yolo_model_path = str(
            params.get(
                "POSE_YOLO_MODEL_DIR",
                params.get("POSE_MODEL_PATH", params.get("YOLO_POSE_MODEL_PATH", "")),
            )
            or ""
        ).strip()
        if pose_model_type == "sleap" and sleap_model_path:
            pose_cfg = PoseConfig(
                backend="sleap",
                sleap=PoseSLEAPConfig(
                    model_path=sleap_model_path,
                    compute_runtime=pose_runtime,
                    batch_size=int(params.get("POSE_BATCH_SIZE", 4)),
                ),
                **common_pose_kwargs,
            )
        elif yolo_model_path and os.path.exists(yolo_model_path):
            pose_cfg = PoseConfig(
                backend="yolo",
                yolo=PoseYOLOConfig(
                    model_path=yolo_model_path,
                    compute_runtime=pose_runtime,
                    confidence_threshold=float(
                        params.get("POSE_CONFIDENCE_THRESHOLD", 1e-4)
                    ),
                    iou_threshold=float(params.get("POSE_IOU_THRESHOLD", 0.7)),
                    max_detections_per_crop=1,
                    batch_size=int(params.get("POSE_BATCH_SIZE", 64)),
                ),
                **common_pose_kwargs,
            )

    apriltag_cfg = AprilTagConfig(
        enabled=bool(params.get("USE_APRILTAGS", False)),
        tag_family=str(params.get("APRILTAG_FAMILY", "tag36h11")),
        threads=int(params.get("APRILTAG_THREADS", 4)),
        max_hamming=int(params.get("APRILTAG_MAX_HAMMING", 1)),
        decimate=float(params.get("APRILTAG_DECIMATE", 1.0)),
        blur=float(params.get("APRILTAG_BLUR", 0.8)),
        crop_padding=float(params.get("INDIVIDUAL_CROP_PADDING", 0.1)),
    )

    batch_size = int(params.get("YOLO_BATCH_SIZE", params.get("BATCH_SIZE", 1)))

    return InferenceConfig(
        obb=obb_cfg,
        headtail=headtail_cfg,
        cnn_phases=cnn_phases,
        pose=pose_cfg,
        apriltag=apriltag_cfg,
        detection_batch_size=batch_size,
        realtime=False,
        use_cache=True,
        runtime_tier=runtime_tier,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_inference_config_from_params.py -q`
Expected: PASS.

- [ ] **Step 5: Update the worker to delegate, and delete its private method**

In `src/hydra_suite/core/tracking/worker.py`:
- At the call site (line 984), replace `self._build_inference_config_from_params(p)` with `build_inference_config_from_params(p)`.
- Add to the worker's imports from `hydra_suite.core.inference.config`: `build_inference_config_from_params` (find the existing `from hydra_suite.core.inference.config import ...` block used for `InferenceConfig`; if config symbols are imported lazily, add the import next to the call site instead).
- Delete the entire `_build_inference_config_from_params` method (lines 4364–4665).

- [ ] **Step 6: Verify the worker still builds its config**

Run:
```bash
grep -rn "_build_inference_config_from_params" src/
python -m pytest tests/ -m "not benchmark" -k "worker and inference" -q
```
Expected: first grep prints nothing; targeted worker tests pass. (If no worker+inference test exists, run `python -c "import hydra_suite.core.tracking.worker"` and confirm no ImportError.)

- [ ] **Step 7: Commit**

```bash
make format
git add -A
git commit -m "refactor(inference): promote build_inference_config_from_params into config.py"
```

---

### Task 3: Add `build_obb_only_config` helper

Later plans migrate `dataset_generation.py` and `detectkit/gui/prediction_preview.py` — both want a detection-only config from a model path + runtime, without assembling a full params dict. Provide a thin wrapper over `build_inference_config_from_params`.

**Files:**
- Modify: `src/hydra_suite/core/inference/config.py`
- Test: `tests/test_inference_config_from_params.py` (extend)

**Interfaces:**
- Consumes: `build_inference_config_from_params` (Task 2).
- Produces: `build_obb_only_config(model_path: str, *, compute_runtime: str = "cpu", confidence_threshold: float = 0.25, iou_threshold: float = 0.7, max_targets: int = 8, mode: str = "direct") -> InferenceConfig`.

- [ ] **Step 1: Write the failing test (append to the Task 2 test file)**

```python
def test_build_obb_only_config_is_detection_only():
    from hydra_suite.core.inference.config import build_obb_only_config

    cfg = build_obb_only_config("m.pt", compute_runtime="cpu", confidence_threshold=0.3)
    assert cfg.obb is not None and cfg.obb.direct.model_path == "m.pt"
    assert cfg.obb.confidence_threshold == 0.3
    assert cfg.headtail is None and cfg.cnn_phases == [] and cfg.pose is None
    assert cfg.apriltag.enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_config_from_params.py::test_build_obb_only_config_is_detection_only -q`
Expected: FAIL — `ImportError: cannot import name 'build_obb_only_config'`.

- [ ] **Step 3: Add the helper to `config.py`**

```python
def build_obb_only_config(
    model_path: str,
    *,
    compute_runtime: str = "cpu",
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    max_targets: int = 8,
    mode: str = "direct",
) -> InferenceConfig:
    """Detection-only InferenceConfig for one-shot / dataset OBB detection.

    Thin wrapper over build_inference_config_from_params with every non-OBB
    stage left disabled. Used by callers that have a model path + runtime but
    no full tracking params dict.
    """
    return build_inference_config_from_params(
        {
            "DETECTION_METHOD": "yolo_obb",
            "YOLO_OBB_MODE": mode,
            "YOLO_OBB_DIRECT_MODEL_PATH": model_path,
            "COMPUTE_RUNTIME": compute_runtime,
            "YOLO_CONFIDENCE_THRESHOLD": confidence_threshold,
            "YOLO_IOU_THRESHOLD": iou_threshold,
            "MAX_TARGETS": max_targets,
        }
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_inference_config_from_params.py -q`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "feat(inference): add build_obb_only_config detection-only helper"
```

---

### Task 4: Fix `predict_pose_for_image` (never-run one-shot pose helper)

`api.py:52` imports `_load_pose_model` from `runner` — that name does not exist (the real one is `load_pose_model` in `stages/pose.py`). And `api.py:109` passes `[image]` (raw images) to `run_pose`, which actually wants a **crops tensor** and returns a **scalar `PoseResult`** (not a list). This function has never executed. Fix the import and the crops flow, mirroring how the runner does pose.

**Files:**
- Modify: `src/hydra_suite/core/inference/api.py:41-113`
- Test: `tests/test_inference_api_pose.py`

**Interfaces:**
- Consumes: `stages.pose.load_pose_model(config, runtime) -> PoseModel`, `stages.pose.run_pose(crops, obb_result, model, config, runtime, aspect_ratio, margin) -> PoseResult`, `stages.crops.extract_canonical_crops(frame, obb_result, ar, mg, runtime, ...) -> torch.Tensor`.
- Produces: `predict_pose_for_image(image, pose_config) -> PoseResult | None` (unchanged public signature).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_api_pose.py
"""predict_pose_for_image must wire crops -> run_pose correctly. It has never
run before this fix (it imported a nonexistent symbol)."""
import types

import numpy as np

import hydra_suite.core.inference.api as api


def test_predict_pose_for_image_wires_crops_to_run_pose(monkeypatch):
    calls = {}

    fake_model = object()
    monkeypatch.setattr(api, "_HAS_RUN", True, raising=False)

    def fake_load_pose_model(cfg, runtime):
        calls["loaded"] = True
        return fake_model

    def fake_extract_canonical_crops(frame, obb, ar, mg, runtime, **kw):
        calls["crops_frame_shape"] = frame.shape
        return "CROPS_TENSOR"

    def fake_run_pose(crops, obb, model, cfg, runtime, ar, mg):
        calls["run_pose_crops"] = crops
        calls["run_pose_model"] = model
        return "POSE_RESULT"  # scalar, not a list

    monkeypatch.setattr(
        "hydra_suite.core.inference.stages.pose.load_pose_model",
        fake_load_pose_model,
    )
    monkeypatch.setattr(
        "hydra_suite.core.inference.stages.pose.run_pose", fake_run_pose
    )
    monkeypatch.setattr(
        "hydra_suite.core.inference.stages.crops.extract_canonical_crops",
        fake_extract_canonical_crops,
    )

    pose_config = types.SimpleNamespace(
        yolo=types.SimpleNamespace(compute_runtime="cpu"),
        sleap=None,
    )
    image = np.zeros((64, 32, 3), dtype=np.uint8)

    result = api.predict_pose_for_image(image, pose_config)

    assert result == "POSE_RESULT"          # scalar returned, not results[0]
    assert calls["run_pose_crops"] == "CROPS_TENSOR"   # crops, not raw [image]
    assert calls["run_pose_model"] is fake_model
    assert calls["loaded"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_api_pose.py -q`
Expected: FAIL — currently `predict_pose_for_image` raises `ImportError` on `from .runner import _load_pose_model`.

- [ ] **Step 3: Rewrite `predict_pose_for_image`**

Replace the body of `predict_pose_for_image` (api.py lines 41–113) with:

```python
def predict_pose_for_image(image, pose_config) -> "PoseResult":  # noqa: F821
    """One-shot pose prediction on a single image, used by PoseKit labeling UI.

    Loads a pose model, builds a whole-image canonical crop, runs pose once,
    and discards the model. NOT for batch use — call InferenceRunner.run_realtime
    if you need persistent state.
    """
    import numpy as np

    from .config import InferenceConfig, OBBConfig, OBBDirectConfig
    from .result import OBBResult
    from .runtime import RuntimeContext
    from .stages.crops import extract_canonical_crops
    from .stages.pose import load_pose_model, run_pose

    compute_runtime = "cpu"
    if pose_config is not None:
        if getattr(pose_config, "yolo", None) is not None:
            compute_runtime = getattr(pose_config.yolo, "compute_runtime", "cpu")
        elif getattr(pose_config, "sleap", None) is not None:
            compute_runtime = getattr(pose_config.sleap, "compute_runtime", "cpu")

    _min_cfg = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="", compute_runtime=compute_runtime),
        ),
        pose=pose_config,
    )
    try:
        runtime = RuntimeContext.from_config(_min_cfg)
    except Exception:
        _min_cfg.obb.direct.compute_runtime = "cpu"
        runtime = RuntimeContext(
            cuda_mode=False,
            device="cpu",
            use_nvdec=False,
            default_runtime="cpu",
            tensor_on_cuda=False,
            requested_gpu=False,
        )

    h, w = image.shape[:2] if hasattr(image, "shape") else (1, 1)
    synthetic_obb = OBBResult(
        frame_idx=0,
        centroids=np.array([[w / 2, h / 2]], dtype=np.float32),
        angles=np.zeros(1, dtype=np.float32),
        sizes=np.array([float(w * h)], dtype=np.float32),
        shapes=np.array([[float(w * h), float(w) / float(h + 1e-6)]], dtype=np.float32),
        confidences=np.ones(1, dtype=np.float32),
        corners=np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.float32),
        detection_ids=OBBResult.make_detection_ids(0, 1),
    )

    ar = 2.0
    mg = 1.3
    model = load_pose_model(pose_config, runtime)
    try:
        crops = extract_canonical_crops(image, synthetic_obb, ar, mg, runtime)
        return run_pose(crops, synthetic_obb, model, pose_config, runtime, ar, mg)
    finally:
        del model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_inference_api_pose.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "fix(inference): repair predict_pose_for_image crops->run_pose wiring"
```

---

### Task 5: Add `InferenceRunner.detect_batch` (batched-in-memory detection)

The dataset-generation worker needs batched OBB detection returning results directly (no cache on disk). `run_realtime` is per-frame; `run_batch_pass` writes to cache and returns `None`. Add one method that mirrors `run_realtime`'s detect+filter prefix but over a list of frames.

**Files:**
- Modify: `src/hydra_suite/core/inference/runner.py`
- Test: `tests/test_inference_runner_detect_batch.py`

**Interfaces:**
- Consumes: `run_obb`, `materialize_tensors`, `_RawOBBTensors` (already imported in runner.py:49), `filter_for_source` (already used at runner.py:495), `OBBResult`.
- Produces: `InferenceRunner.detect_batch(frames: list[np.ndarray], frame_indices: list[int] | None = None, roi_mask=None) -> list[OBBResult]` (filtered OBB results, one per frame).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_runner_detect_batch.py
"""detect_batch returns filtered OBBResults in memory, mirroring run_realtime's
detect+filter prefix, without touching any cache."""
import numpy as np

from hydra_suite.core.inference.runner import InferenceRunner


class _FakeOBBModels:
    def close(self):
        pass


def _make_runner_with_fakes(monkeypatch, per_frame_counts):
    # Bypass __init__ so no real models load.
    runner = InferenceRunner.__new__(InferenceRunner)
    runner.config = _make_obb_only_config()
    runner._models = type("M", (), {"obb": _FakeOBBModels(), "bgsub": None})()
    runner.runtime = object()
    runner._caches = None

    import hydra_suite.core.inference.runner as rmod
    from hydra_suite.core.inference.result import OBBResult

    def fake_run_obb(frames, models, cfg, runtime):
        out = []
        for n in per_frame_counts[: len(frames)]:
            out.append(
                OBBResult(
                    frame_idx=0,
                    centroids=np.zeros((n, 2), np.float32),
                    angles=np.zeros(n, np.float32),
                    sizes=np.ones(n, np.float32),
                    shapes=np.ones((n, 2), np.float32),
                    confidences=np.ones(n, np.float32),
                    corners=np.zeros((n, 4, 2), np.float32),
                    detection_ids=OBBResult.make_detection_ids(0, n),
                )
            )
        return out

    def fake_filter_for_source(config, raw_obb, roi_mask=None):
        return raw_obb, np.arange(raw_obb.num_detections, dtype=np.int32)

    monkeypatch.setattr(rmod, "run_obb", fake_run_obb)
    monkeypatch.setattr(rmod, "filter_for_source", fake_filter_for_source)
    return runner


def _make_obb_only_config():
    from hydra_suite.core.inference.config import build_obb_only_config

    return build_obb_only_config("m.pt", compute_runtime="cpu")


def test_detect_batch_returns_one_result_per_frame_with_frame_idx(monkeypatch):
    runner = _make_runner_with_fakes(monkeypatch, per_frame_counts=[3, 0, 5])
    frames = [np.zeros((8, 8, 3), np.uint8) for _ in range(3)]
    results = runner.detect_batch(frames, frame_indices=[10, 11, 12])
    assert [r.num_detections for r in results] == [3, 0, 5]
    assert [r.frame_idx for r in results] == [10, 11, 12]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_runner_detect_batch.py -q`
Expected: FAIL — `AttributeError: 'InferenceRunner' object has no attribute 'detect_batch'`.

- [ ] **Step 3: Add the method to `InferenceRunner`**

Add this method to the `InferenceRunner` class (e.g. directly after `run_realtime`). It reuses the exact detect+filter sequence from `run_realtime` (runner.py:467–495):

```python
    def detect_batch(
        self,
        frames: "list[np.ndarray]",
        frame_indices: "list[int] | None" = None,
        roi_mask: "np.ndarray | None" = None,
    ) -> "list[OBBResult]":
        """Run OBB detection over a list of frames, returning filtered results
        in memory. No cache is read or written. Mirrors run_realtime's
        detect+filter prefix; for the dataset-generation batched path.
        """
        if self._models.obb is None:
            raise RuntimeError("detect_batch requires an OBB detection config (config.obb)")
        frames = list(frames)
        if frame_indices is None:
            frame_indices = list(range(len(frames)))

        raw_list = run_obb(frames, self._models.obb, self.config.obb, self.runtime)
        results: list[OBBResult] = []
        for raw, f_idx in zip(raw_list, frame_indices):
            if isinstance(raw, _RawOBBTensors):
                raw_obb = materialize_tensors(raw, self.config.obb.raw_detection_cap)
            else:
                raw_obb = raw
            raw_obb = OBBResult(
                frame_idx=f_idx,
                centroids=raw_obb.centroids,
                angles=raw_obb.angles,
                sizes=raw_obb.sizes,
                shapes=raw_obb.shapes,
                confidences=raw_obb.confidences,
                corners=raw_obb.corners,
                detection_ids=OBBResult.make_detection_ids(f_idx, raw_obb.num_detections),
            )
            filtered_obb, _ = filter_for_source(self.config, raw_obb, roi_mask)
            results.append(filtered_obb)
        return results
```

Confirm `OBBResult` is importable at module scope in `runner.py` (it is used by `run_realtime`); if it is only imported locally there, add `from .result import OBBResult` to the module imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_inference_runner_detect_batch.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "feat(inference): add InferenceRunner.detect_batch batched-in-memory detection"
```

---

## Final verification (whole plan)

- [ ] **Step 1: Full test suite**

Run: `python -m pytest tests/ -m "not benchmark" -q`
Expected: PASS (no regressions). Pay attention that `tests/test_inference_obb_artifacts.py` still passes.

- [ ] **Step 2: Confirm core/inference is detector-free**

Run:
```bash
grep -rn "core.detectors\|core/detectors\|_direct_obb_runtime" src/hydra_suite/core/inference/
```
Expected: no output.

- [ ] **Step 3: Confirm the worker delegates**

Run:
```bash
grep -rn "_build_inference_config_from_params" src/
python -c "import hydra_suite.core.tracking.worker"
```
Expected: no matches; no ImportError.

- [ ] **Step 4: Format gate**

Run: `make format-check` (or `make format` then confirm clean `git status`).

---

## Self-Review notes

- **Spec coverage (Phases A + B):** Task 1 = relocate `_direct_obb_runtime` + fix docstring (Phase A). Tasks 2–3 = promote config builder + OBB-only helper (Phase B bullet 1). Task 4 = fix `api.py` (Phase B bullet 2). Task 5 = batched-in-memory method (Phase B bullet 3). Phase A's "repoint tools scripts" is Task 1 Step 5.
- **Deferred to later plans (correctly out of scope here):** all consumer migrations (Phase C → Plan 2), benchmarking (Phase D → Plan 3), deletion of `core/detectors` + test migration + parity gate (Phases E/F → Plan 4). `core/detectors` still exists and is still used by consumers after this plan — that is expected; Plan 1 only removes the `core/inference → core/detectors` edge.
- **Type consistency:** `build_inference_config_from_params(params) -> InferenceConfig` and `build_obb_only_config(...) -> InferenceConfig` used consistently; `detect_batch(...) -> list[OBBResult]`; `predict_pose_for_image(image, pose_config) -> PoseResult | None`. `run_pose` called with the 7-positional-arg form `(crops, obb, model, config, runtime, ar, mg)` matching `stages/pose.py:162`.
