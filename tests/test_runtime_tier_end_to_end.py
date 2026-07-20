"""End-to-end integration tests for legacy-config → runtime_tier migration.

These tests write a minimal LEGACY-format config (with per-stage
``compute_runtime`` strings and NO top-level ``runtime_tier``) to a JSON
file, load it via ``InferenceConfig.from_json``, and assert that the derived
``runtime_tier`` is correct.  They act as a regression guard for the
migration path introduced in Phase 2.
"""

import json

from hydra_suite.core.inference.config import InferenceConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_OBB_CPU = {
    "mode": "direct",
    "direct": {"model_path": "/tmp/obb.pt", "compute_runtime": "cpu"},
}


def _write_and_load(tmp_path, payload: dict) -> InferenceConfig:
    """Write *payload* to a temp JSON file and load via InferenceConfig.from_json."""
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(payload))
    return InferenceConfig.from_json(str(p))


# ---------------------------------------------------------------------------
# Core migration cases
# ---------------------------------------------------------------------------


def test_legacy_tensorrt_obb_direct_migrates_to_gpu_fast(tmp_path):
    """obb.direct.compute_runtime='tensorrt' → runtime_tier == 'gpu_fast'."""
    payload = {
        "obb": {
            "mode": "direct",
            "direct": {"model_path": "m.pt", "compute_runtime": "tensorrt"},
        },
        "pipeline_depth": 2,
    }
    cfg = _write_and_load(tmp_path, payload)
    assert (
        cfg.runtime_tier == "gpu_fast"
    ), f"Expected 'gpu_fast' from tensorrt legacy config, got {cfg.runtime_tier!r}"


def test_legacy_all_cpu_migrates_to_cpu(tmp_path):
    """All-cpu per-stage runtimes → runtime_tier == 'cpu'."""
    payload = {
        "obb": {
            "mode": "direct",
            "direct": {"model_path": "m.pt", "compute_runtime": "cpu"},
        },
        "headtail": {"model_path": "ht.pt", "compute_runtime": "cpu"},
        "cnn_phases": [
            {"label": "identity", "model_path": "cnn.pt", "compute_runtime": "cpu"}
        ],
    }
    cfg = _write_and_load(tmp_path, payload)
    assert (
        cfg.runtime_tier == "cpu"
    ), f"Expected 'cpu' from all-cpu legacy config, got {cfg.runtime_tier!r}"


def test_legacy_cuda_obb_migrates_to_gpu(tmp_path):
    """obb.direct.compute_runtime='cuda' → runtime_tier == 'gpu'."""
    payload = {
        "obb": {
            "mode": "direct",
            "direct": {"model_path": "m.pt", "compute_runtime": "cuda"},
        },
    }
    cfg = _write_and_load(tmp_path, payload)
    assert (
        cfg.runtime_tier == "gpu"
    ), f"Expected 'gpu' from cuda legacy config, got {cfg.runtime_tier!r}"


def test_legacy_mps_obb_migrates_to_gpu(tmp_path):
    """obb.direct.compute_runtime='mps' → runtime_tier == 'gpu'."""
    payload = {
        "obb": {
            "mode": "direct",
            "direct": {"model_path": "m.pt", "compute_runtime": "mps"},
        },
    }
    cfg = _write_and_load(tmp_path, payload)
    assert (
        cfg.runtime_tier == "gpu"
    ), f"Expected 'gpu' from mps legacy config, got {cfg.runtime_tier!r}"


# ---------------------------------------------------------------------------
# Pose-stage migration (regression guard for Task 2 ordering fix)
# ---------------------------------------------------------------------------


def test_legacy_pose_yolo_tensorrt_migrates_to_gpu_fast(tmp_path):
    """pose.yolo.compute_runtime='tensorrt' → runtime_tier == 'gpu_fast'.

    Regression guard: the migration path must inspect pose sub-configs before
    constructing PoseConfig objects (ordering fix from Task 2).
    """
    payload = {
        "obb": _MINIMAL_OBB_CPU,
        "pose": {
            "yolo": {"model_path": "pose.pt", "compute_runtime": "tensorrt"},
        },
    }
    # NOTE: this config mixes 'tensorrt' with 'cpu' on obb.direct, a legacy
    # per-stage combination that is no longer validated (Gen-2 collapses all
    # stages to one runtime_tier). We test only the tier derivation path via
    # _dict_to_config directly here.
    from hydra_suite.core.inference.config import _dict_to_config

    cfg = _dict_to_config(payload)
    assert (
        cfg.runtime_tier == "gpu_fast"
    ), f"Expected 'gpu_fast' from pose.yolo.tensorrt, got {cfg.runtime_tier!r}"


def test_legacy_pose_sleap_onnx_cuda_migrates_to_gpu_fast(tmp_path):
    """pose.sleap.compute_runtime='onnx_cuda' → runtime_tier == 'gpu_fast'.

    Regression guard for the pose-migration ordering fix from Task 2.
    """
    from hydra_suite.core.inference.config import _dict_to_config

    payload = {
        "obb": _MINIMAL_OBB_CPU,
        "pose": {
            "sleap": {"model_path": "sleap.pt", "compute_runtime": "onnx_cuda"},
        },
    }
    cfg = _dict_to_config(payload)
    assert (
        cfg.runtime_tier == "gpu_fast"
    ), f"Expected 'gpu_fast' from pose.sleap.onnx_cuda, got {cfg.runtime_tier!r}"


def test_legacy_pose_yolo_cuda_with_obb_cuda_full_roundtrip(tmp_path):
    """Full from_json round-trip: cuda pose + cuda obb → runtime_tier == 'gpu'."""
    payload = {
        "obb": {
            "mode": "direct",
            "direct": {"model_path": "m.pt", "compute_runtime": "cuda"},
        },
        "pose": {
            "yolo": {"model_path": "pose.pt", "compute_runtime": "cuda"},
        },
    }
    cfg = _write_and_load(tmp_path, payload)
    assert (
        cfg.runtime_tier == "gpu"
    ), f"Expected 'gpu' from all-cuda config, got {cfg.runtime_tier!r}"


# ---------------------------------------------------------------------------
# Sequential OBB migration
# ---------------------------------------------------------------------------


def test_legacy_sequential_tensorrt_migrates_to_gpu_fast(tmp_path):
    """obb.sequential with tensorrt obb_compute_runtime → runtime_tier == 'gpu_fast'."""
    payload = {
        "obb": {
            "mode": "sequential",
            "sequential": {
                "detect_model_path": "det.pt",
                "obb_model_path": "obb.pt",
                "detect_compute_runtime": "cuda",
                "obb_compute_runtime": "tensorrt",
            },
        },
    }
    cfg = _write_and_load(tmp_path, payload)
    assert (
        cfg.runtime_tier == "gpu_fast"
    ), f"Expected 'gpu_fast' from sequential tensorrt, got {cfg.runtime_tier!r}"


# ---------------------------------------------------------------------------
# Explicit runtime_tier in config is preserved (not overwritten by migration)
# ---------------------------------------------------------------------------


def test_explicit_runtime_tier_is_preserved(tmp_path):
    """When top-level runtime_tier is present, migration is skipped."""
    payload = {
        "obb": {
            "mode": "direct",
            "direct": {"model_path": "m.pt", "compute_runtime": "cpu"},
        },
        "runtime_tier": "gpu_fast",
    }
    cfg = _write_and_load(tmp_path, payload)
    assert (
        cfg.runtime_tier == "gpu_fast"
    ), f"Explicit runtime_tier should be preserved, got {cfg.runtime_tier!r}"
