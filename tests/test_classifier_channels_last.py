import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hydra_suite.core.identity.classification import backend as backend_mod


class _TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):
        x = self.pool(self.conv(x)).flatten(1)
        return self.fc(x)


def _make_backend(monkeypatch, device_str):
    be = backend_mod.ClassifierBackend.__new__(backend_mod.ClassifierBackend)
    be._compute_runtime = "cuda" if device_str == "cuda" else "cpu"
    be._model = _TinyNet().eval()
    monkeypatch.setattr(backend_mod, "_torch_device", lambda rt: device_str)
    return be


def test_forward_torch_cpu_stays_contiguous(monkeypatch):
    be = _make_backend(monkeypatch, "cpu")
    batch = np.random.rand(2, 3, 8, 8).astype(np.float32)
    out = be._forward_torch(batch)
    assert out.shape == (2, 2)
    # CPU model params remain default (contiguous) memory format
    assert (
        not be._model.conv.weight.is_contiguous(memory_format=torch.channels_last)
        or be._model.conv.weight.is_contiguous()
    )  # unchanged on CPU


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_forward_torch_cuda_uses_channels_last(monkeypatch):
    be = _make_backend(monkeypatch, "cuda")
    be._model = be._model.cuda()
    batch = np.random.rand(2, 3, 8, 8).astype(np.float32)
    out = be._forward_torch(batch)
    assert out.shape == (2, 2)
    assert be._model.conv.weight.is_contiguous(memory_format=torch.channels_last)
