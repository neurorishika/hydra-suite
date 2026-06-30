"""Tests for RuntimeContext stream-sync chokepoint (Task 9).

The CUDA event path is validated separately on a CUDA box.
These tests cover:
  - CPU identity: handoff/await_handoff are no-ops that return the same tensor
  - No surprising state attached to the tensor on CPU
  - await_handoff on a tensor that never went through handoff is a safe no-op
"""

import pytest
import torch

from hydra_suite.core.inference.runtime import _HANDOFF_EVENTS, RuntimeContext

_CPU_RT = RuntimeContext(
    cuda_mode=False,
    device="cpu",
    use_nvdec=False,
    tensor_on_cuda=False,
    default_runtime="cpu",
)


def test_handoff_is_identity_on_cpu():
    rt = _CPU_RT
    t = torch.arange(6)
    assert rt.await_handoff(rt.handoff(t)) is t


def test_handoff_does_not_attach_state_on_cpu():
    """On CPU, handoff must not put the tensor into the event map."""
    t = torch.arange(4)
    _CPU_RT.handoff(t)
    # The module-level WeakKeyDictionary must not have an entry for a CPU tensor
    assert t not in _HANDOFF_EVENTS


def test_await_handoff_without_prior_handoff_is_safe():
    """await_handoff on a tensor that was never handed off must be a no-op."""
    t = torch.zeros(3)
    result = _CPU_RT.await_handoff(t)
    assert result is t


def test_handoff_returns_tensor_unchanged_on_cpu():
    """The returned tensor must be the exact same object (not a copy)."""
    t = torch.ones(5)
    out = _CPU_RT.handoff(t)
    assert out is t
