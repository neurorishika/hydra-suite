"""Tests for CoreML .mlpackage exporters (tiny + torchvision/timm classifiers).

These tests are skipped automatically when coremltools is not installed.
On Apple Silicon machines with coremltools present the tests perform a real
export and verify that the .mlpackage bundle is produced on disk.
"""

import pytest

pytest.importorskip("coremltools")
torch = pytest.importorskip("torch")

from hydra_suite.training.tiny_model import export_tiny_to_coreml  # noqa: E402
from hydra_suite.training.torchvision_model import (  # noqa: E402
    export_torchvision_to_coreml,
)


class _Net(torch.nn.Module):
    """Minimal conv-net used as a stand-in for both classifier families."""

    def __init__(self):
        super().__init__()
        self.c = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.p = torch.nn.AdaptiveAvgPool2d(1)
        self.f = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.f(self.p(self.c(x)).flatten(1))


def test_export_torchvision_to_coreml_writes_mlpackage(tmp_path):
    """export_torchvision_to_coreml must produce a real .mlpackage on disk."""
    out = tmp_path / "m.mlpackage"
    p = export_torchvision_to_coreml(_Net().eval(), {"input_size": (32, 32)}, out)
    assert p.exists(), f".mlpackage not found at {p}"
    assert str(p).endswith(".mlpackage")


def test_export_tiny_to_coreml_writes_mlpackage(tmp_path):
    """export_tiny_to_coreml must produce a real .mlpackage on disk."""
    out = tmp_path / "tiny.mlpackage"
    p = export_tiny_to_coreml(_Net().eval(), {"input_size": [32, 32]}, out)
    assert p.exists(), f".mlpackage not found at {p}"
    assert str(p).endswith(".mlpackage")


def test_export_torchvision_to_coreml_default_input_size(tmp_path):
    """Default input_size (224, 224) must be used when ckpt has no input_size."""
    out = tmp_path / "default.mlpackage"
    p = export_torchvision_to_coreml(_Net().eval(), {}, out)
    assert p.exists()
    assert str(p).endswith(".mlpackage")


def test_export_tiny_to_coreml_default_input_size(tmp_path):
    """Default input_size [64, 128] must be used when ckpt has no input_size."""
    out = tmp_path / "tiny_default.mlpackage"
    p = export_tiny_to_coreml(_Net().eval(), {}, out)
    assert p.exists()
    assert str(p).endswith(".mlpackage")
