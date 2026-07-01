"""Test: best-effort channels_last memory format for torch OBB executor on CUDA.

The conversion is CUDA-only and best-effort (wrapped in try/except).
On MPS/CPU hosts this test is skipped cleanly — no model download needed.
"""

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_obb_torch_executor_cuda_channels_last(tmp_path):
    from ultralytics import YOLO

    from hydra_suite.core.inference.runtime_artifacts import load_obb_executor

    pt = str(tmp_path / "yolov8n-obb.pt")
    YOLO("yolov8n-obb.pt").save(pt)  # ultralytics resolves/downloads the base model
    exe = load_obb_executor(pt, compute_runtime="cuda", auto_export=False, max_det=100)
    # The wrapped torch model should be in channels_last on CUDA. Assert on a 4D
    # conv weight — 1D tensors (biases) are vacuously contiguous in every format,
    # so `next(...parameters())` alone could pass without the conversion.
    conv_w = next(t for t in exe.model.parameters() if t.dim() == 4)
    assert conv_w.is_contiguous(memory_format=torch.channels_last)
