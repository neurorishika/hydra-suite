"""Tests for RuntimeContext stream-sync chokepoint (Task 9).

The CUDA event path is validated separately on a CUDA box.
These tests cover:
  - CPU identity: handoff/await_handoff are no-ops that return the same tensor
  - No surprising state attached to the tensor on CPU
  - await_handoff on a tensor that never went through handoff is a safe no-op
"""

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
    # The module-level event map (keyed by id(tensor)) must have no entry for a
    # CPU tensor — the CUDA branch is never taken on a CPU RuntimeContext.
    assert id(t) not in _HANDOFF_EVENTS


def test_handoff_keying_survives_multielement_tensor(monkeypatch):
    """Regression (found on mehek/CUDA): the handoff event map MUST be keyed by
    ``id(tensor)``, never the tensor itself. A torch tensor as a dict key raises
    "Boolean value of Tensor with more than one value is ambiguous" on lookup,
    because ``Tensor.__eq__`` returns an element-wise tensor. This simulates the
    CUDA handoff path on a CPU host by faking ``torch.cuda`` so the keying logic
    is exercised without a GPU; with the old WeakKeyDictionary keyed by tensor
    this raised, with id-keying it does not.
    """
    from hydra_suite.core.inference import runtime as rt_mod

    class _FakeEvent:
        def record(self, *a, **k):
            pass

    class _FakeStream:
        def wait_event(self, *a, **k):
            pass

    monkeypatch.setattr(torch.cuda, "Event", lambda *a, **k: _FakeEvent())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: _FakeStream())

    rt = RuntimeContext(
        cuda_mode=True,
        device="cuda:0",
        use_nvdec=False,
        tensor_on_cuda=True,
        default_runtime="cuda",
        requested_gpu=True,
    )
    t = torch.arange(12).reshape(3, 4)  # multi-element — the exact trap
    assert rt.handoff(t) is t
    assert id(t) in rt_mod._HANDOFF_EVENTS  # recorded under id, not the tensor
    # The lookup must NOT raise on a multi-element tensor, and must pop the entry.
    assert rt.await_handoff(t) is t
    assert id(t) not in rt_mod._HANDOFF_EVENTS


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
