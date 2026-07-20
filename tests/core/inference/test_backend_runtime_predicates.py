import pytest

from hydra_suite.core.identity.classification.backend import _torch_device_for_resolved
from hydra_suite.runtime.resolver import ResolvedBackend

FIVE = [
    (ResolvedBackend("torch", "cpu", False), "cpu"),
    (ResolvedBackend("torch", "mps", False), "mps"),
    (ResolvedBackend("torch", "cuda", False), "cuda"),
    (ResolvedBackend("tensorrt", "cuda", False), "cuda"),
    (ResolvedBackend("coreml", "mps", False), "mps"),
]


@pytest.mark.parametrize("resolved,expected_device", FIVE)
def test_torch_device_for_resolved(resolved, expected_device):
    assert _torch_device_for_resolved(resolved) == expected_device
