from __future__ import annotations

import sys
from types import SimpleNamespace

from tests.helpers.module_loader import load_src_module, make_cv2_stub

batch_mod = load_src_module(
    "hydra_suite/utils/batch_optimizer.py",
    "hydra_suite.utils.batch_optimizer_under_test",
    stubs={"cv2": make_cv2_stub()},
)
BatchOptimizer = batch_mod.BatchOptimizer


def test_batch_optimizer_manual_mode_clamps_to_tensorrt_limit() -> None:
    optimizer = BatchOptimizer(
        {
            "yolo_batch_size_mode": "manual",
            "yolo_manual_batch_size": 128,
            "enable_tensorrt": True,
            "tensorrt_max_batch_size": 16,
        }
    )
    optimizer.device_type = "cuda"
    optimizer.available_memory = 12000
    batch = optimizer.estimate_batch_size(1920, 1080, "yolo26s-obb.pt")
    assert batch == 16


def test_batch_optimizer_cpu_always_returns_one() -> None:
    optimizer = BatchOptimizer({})
    optimizer.device_type = "cpu"
    optimizer.available_memory = 0
    batch = optimizer.estimate_batch_size(1280, 720, "yolo26n-obb.pt")
    assert batch == 1


def test_batch_optimizer_auto_mode_returns_positive_batch() -> None:
    optimizer = BatchOptimizer(
        {
            "enable_yolo_batching": True,
            "cuda_memory_fraction": 0.7,
            "mps_memory_fraction": 0.3,
        }
    )
    optimizer.device_type = "cuda"
    optimizer.device_name = "Synthetic CUDA"
    optimizer.available_memory = 16000
    batch = optimizer.estimate_batch_size(1280, 720, "yolo26s-obb.pt")
    assert batch >= 1


def test_batch_optimizer_realtime_forces_frame_batch_one() -> None:
    optimizer = BatchOptimizer(
        {
            "enable_yolo_batching": True,
            "yolo_batch_size_mode": "manual",
            "yolo_manual_batch_size": 32,
            "tracking_realtime_mode": True,
            "tracking_workflow_mode": "realtime",
        }
    )
    optimizer.device_type = "mps"
    optimizer.available_memory = 16000

    batch = optimizer.estimate_batch_size(1280, 720, "yolo26s-obb.pt")

    assert batch == 1


def test_mps_unified_memory_fraction_is_not_applied_during_detection(
    monkeypatch,
) -> None:
    monkeypatch.setattr(batch_mod, "TORCH_CUDA_AVAILABLE", False)
    monkeypatch.setattr(batch_mod, "MPS_AVAILABLE", True)
    monkeypatch.setattr(batch_mod, "torch", object())
    fake_psutil = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=10 * 1024**3)
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    optimizer = BatchOptimizer({"mps_memory_fraction": 0.3})

    optimizer.detect_device()

    assert optimizer.available_memory == 10 * 1024
