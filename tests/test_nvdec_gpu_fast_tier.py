"""Unit tests for NVDEC confined to the gpu_fast tier (spec 2026-07-22-nvdec-gpu-fast-tier).

Runs on the Mac dev box: the tier-gating logic is pure and does not need a CUDA
device. The end-to-end NVDEC-engaged gate is Task 6 (mehek).
"""


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
