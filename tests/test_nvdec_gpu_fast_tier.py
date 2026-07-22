"""Unit tests for NVDEC confined to the gpu_fast tier (spec 2026-07-22-nvdec-gpu-fast-tier).

Runs on the Mac dev box: the tier-gating logic is pure and does not need a CUDA
device. The end-to-end NVDEC-engaged gate is Task 6 (mehek).
"""

from unittest.mock import MagicMock

import torch


def test_should_use_nvdec_gpu_fast_only(monkeypatch):
    from hydra_suite.core.inference import runtime as rt_mod

    # NVDEC libraries present: only gpu_fast enables it.
    monkeypatch.setattr(rt_mod, "_nvdec_available", lambda: True)
    assert rt_mod._should_use_nvdec("gpu_fast") is True
    assert rt_mod._should_use_nvdec("gpu") is False
    assert rt_mod._should_use_nvdec("cpu") is False

    # NVDEC libraries absent: never, even on gpu_fast.
    monkeypatch.setattr(rt_mod, "_nvdec_available", lambda: False)
    assert rt_mod._should_use_nvdec("gpu_fast") is False
    assert rt_mod._should_use_nvdec("gpu") is False


def test_run_direct_gpu_fast_tensorrt_takes_frames_list_path(monkeypatch):
    """gpu_fast: a DirectExecutorAdapter (TensorRT) must receive the raw CUDA
    frame list (its predict does its own letterbox); the manual GPU-letterbox
    pre-batch must be skipped, and the return must be OBBResults (tensor_on_cuda
    False), not raw tensors."""

    from hydra_suite.core.inference.runtime_artifacts import DirectExecutorAdapter
    from hydra_suite.core.inference.stages import obb as obb_stage

    # A frame that passes isinstance(_, torch.Tensor) and reports is_cuda True.
    fake_frame = MagicMock(spec=torch.Tensor)
    fake_frame.is_cuda = True

    model = MagicMock(spec=DirectExecutorAdapter)
    model.predict.return_value = ["raw_result_0"]

    letterbox_spy = {"called": False}
    monkeypatch.setattr(
        obb_stage,
        "_gpu_letterbox_batch",
        lambda *a, **k: letterbox_spy.__setitem__("called", True) or (None, []),
    )
    # Bypass real Ultralytics Results parsing.
    monkeypatch.setattr(obb_stage, "extract_obb_result", lambda r, idx: f"obb::{r}")
    monkeypatch.setattr(obb_stage, "_apply_raw_detection_cap", lambda res, cap: res)

    cfg = type(
        "C",
        (),
        {
            "target_classes": None,
            "raw_detection_cap": 0,
            "direct": type("D", (), {"confidence_floor": 1e-3})(),
        },
    )()
    rt = type("RT", (), {"tensor_on_cuda": False, "device": "cuda"})()

    out = obb_stage._run_direct([fake_frame], model, cfg, rt)

    assert letterbox_spy["called"] is False  # frames-list path, no pre-batch
    passed_frames = model.predict.call_args.args[0]
    assert passed_frames == [fake_frame]  # raw CUDA frame list handed through
    assert out == ["obb::raw_result_0"]  # OBBResult path, not raw tensors
