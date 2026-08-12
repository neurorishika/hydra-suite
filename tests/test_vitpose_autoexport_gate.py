import pytest

from hydra_suite.core.individual.pose.api import create_pose_backend_from_config
from hydra_suite.core.individual.pose.backends import vitpose as vitpose_backend_module
from hydra_suite.core.individual.pose.types import PoseRuntimeConfig
from hydra_suite.core.inference.config import PoseViTPoseConfig


def test_auto_export_defaults_true_and_roundtrips():
    c = PoseViTPoseConfig(model_path="m.pt")
    assert c.auto_export is True
    from dataclasses import asdict

    d = asdict(c)
    assert d["auto_export"] is True
    c2 = PoseViTPoseConfig(**d)
    assert c2.auto_export is True


def _failing_export(config, runtime_flavor, runtime_device=None):
    raise RuntimeError("simulated export failure")


class _StubViTPoseBackend:
    """Records the runtime_flavor it was constructed with, no real checkpoint load."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_auto_export_gate_raises_when_disabled(monkeypatch):
    """auto_export_vitpose_model failing + vitpose_auto_export=False must raise
    loudly (fail-loud gate), not silently fall back to native torch."""
    monkeypatch.setattr(
        vitpose_backend_module, "auto_export_vitpose_model", _failing_export
    )
    monkeypatch.setattr(vitpose_backend_module, "ViTPoseBackend", _StubViTPoseBackend)

    config = PoseRuntimeConfig(
        backend_family="vitpose",
        runtime_flavor="tensorrt",
        model_path="fake_checkpoint.pt",
        vitpose_auto_export=False,
    )

    with pytest.raises(RuntimeError, match="auto_export is disabled"):
        create_pose_backend_from_config(config)


def test_auto_export_gate_falls_back_when_enabled(monkeypatch):
    """Same failing export, but vitpose_auto_export=True (the default) must NOT
    raise -- it should fall back to the native runtime instead, proving the
    gate is conditional rather than always-on."""
    monkeypatch.setattr(
        vitpose_backend_module, "auto_export_vitpose_model", _failing_export
    )
    monkeypatch.setattr(vitpose_backend_module, "ViTPoseBackend", _StubViTPoseBackend)

    config = PoseRuntimeConfig(
        backend_family="vitpose",
        runtime_flavor="tensorrt",
        model_path="fake_checkpoint.pt",
        vitpose_auto_export=True,
    )

    backend = create_pose_backend_from_config(config)

    assert isinstance(backend, _StubViTPoseBackend)
    assert backend.kwargs["runtime_flavor"] == "native"
