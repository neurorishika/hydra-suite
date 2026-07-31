import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hydra_suite.core.canonicalization.crop import (
    gpu_canonical_crop,
    gpu_canonical_crop_batch,
)


def _affine_identity():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def test_gpu_crop_batch_has_no_autograd_graph():
    frame = torch.rand(3, 32, 32, requires_grad=True)
    out = gpu_canonical_crop_batch(
        frame, [_affine_identity(), _affine_identity()], 16, 16
    )
    assert out.shape == (2, 3, 16, 16)
    assert out.grad_fn is None
    assert out.requires_grad is False


def test_gpu_crop_single_has_no_autograd_graph():
    frame = torch.rand(3, 32, 32, requires_grad=True)
    out = gpu_canonical_crop(frame, _affine_identity(), 16, 16)
    assert out.grad_fn is None
    assert out.requires_grad is False
