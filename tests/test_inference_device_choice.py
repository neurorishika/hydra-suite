"""Device selection for SAM2/SAM3 escalation (Auto/CUDA/MPS/CPU on every host)."""

from __future__ import annotations

from types import SimpleNamespace

from hydra_suite.core.inference import torch_device


def test_device_choices_list_only_available_accelerators(monkeypatch):
    monkeypatch.setattr(torch_device, "TORCH_CUDA_AVAILABLE", False)
    monkeypatch.setattr(torch_device, "MPS_AVAILABLE", True)
    values = [v for _label, v in torch_device.device_choices()]
    assert values == ["auto", "mps", "cpu"]


def test_device_choices_cpu_only_host(monkeypatch):
    monkeypatch.setattr(torch_device, "TORCH_CUDA_AVAILABLE", False)
    monkeypatch.setattr(torch_device, "MPS_AVAILABLE", False)
    assert [v for _l, v in torch_device.device_choices()] == ["auto", "cpu"]


def test_unavailable_preference_falls_back(monkeypatch):
    monkeypatch.setattr(torch_device, "TORCH_CUDA_AVAILABLE", False)
    monkeypatch.setattr(torch_device, "MPS_AVAILABLE", False)
    assert torch_device.resolve_torch_device("cuda") == "cpu"
    assert torch_device.resolve_torch_device("cpu") == "cpu"


def test_sam2_worker_builds_executor_on_requested_device(monkeypatch):
    from hydra_suite.core.inference.sam2 import executor as executor_mod
    from hydra_suite.detectkit.jobs import sam2_escalation

    seen = {}

    def fake_from_variant(variant, device=None, **_kw):
        seen.update(variant=variant, device=device)
        raise RuntimeError("stop after construction")

    monkeypatch.setattr(
        executor_mod.Sam2SegmentExecutor,
        "from_variant",
        staticmethod(fake_from_variant),
    )
    request = sam2_escalation.EscalationRequest(
        project=SimpleNamespace(project_dir="/tmp", sources=[]),
        source_names=[],
        variant="sam2.1-hiera-tiny",
        device="cpu",
    )
    worker = sam2_escalation.Sam2EscalationWorker(request)
    try:
        worker.execute()
    except RuntimeError:
        pass
    assert seen == {"variant": "sam2.1-hiera-tiny", "device": "cpu"}


def test_semantic_request_carries_device_outside_the_fingerprint(tmp_path):
    from hydra_suite.detectkit.jobs import semantic_escalation as se

    base = dict(
        project=SimpleNamespace(project_dir=str(tmp_path), sources=[]),
        source_names=[],
        variant="sam3",
        prompt="ant",
    )
    a = se._fingerprint(
        se.SemanticEscalationRequest(**base, device="cpu"), tmp_path, None, 0.1
    )
    b = se._fingerprint(
        se.SemanticEscalationRequest(**base, device="mps"), tmp_path, None, 0.1
    )
    assert a == b


def test_device_combo_round_trip():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from hydra_suite.widgets.device_combo import DeviceCombo

    _app = QApplication.instance() or QApplication([])  # noqa: F841

    combo = DeviceCombo("cpu")
    assert combo.device() == "cpu"
    combo.set_device("not-a-device")
    assert combo.device() == "auto"
