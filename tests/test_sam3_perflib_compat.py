"""The eager replacement for SAM3's inference-only fused MLP kernel."""

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from hydra_suite.training.sam3_lora.perflib_compat import (  # noqa: E402
    eager_addmm_act,
    install_grad_safe_addmm_act,
)


class _Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = torch.nn.Linear(4, 3)
        self.weight = self.inner.weight
        self.bias = self.inner.bias

    def forward(self, x):
        return self.inner(x)


class _NoWeightWrapper(torch.nn.Module):
    """Stands in for LoraLinear: callable, but exposes no `.weight`."""

    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(4, 3)
        self.extra = torch.nn.Parameter(torch.ones(3))

    def forward(self, x):
        return self.base(x) + self.extra


@pytest.mark.parametrize(
    "act, fn",
    [
        (torch.nn.GELU, torch.nn.functional.gelu),
        (torch.nn.ReLU, torch.nn.functional.relu),
    ],
)
def test_eager_addmm_act_matches_the_reference_computation(act, fn):
    linear = _Linear()
    x = torch.randn(2, 5, 4)
    expected = fn(torch.nn.functional.linear(x, linear.weight, linear.bias))
    torch.testing.assert_close(eager_addmm_act(act, linear, x), expected)


def test_eager_addmm_act_propagates_gradient_to_the_weight():
    """The whole point: the fused kernel detaches, so fc1 could never learn."""
    linear = _Linear()
    x = torch.randn(2, 4)
    eager_addmm_act(torch.nn.GELU, linear, x).sum().backward()
    assert linear.weight.grad is not None
    assert torch.any(linear.weight.grad != 0)


def test_eager_addmm_act_works_with_grad_enabled():
    """The fused kernel raises ValueError("Expected grad to be disabled.")."""
    linear = _Linear()
    assert torch.is_grad_enabled()
    eager_addmm_act(torch.nn.GELU, linear, torch.randn(2, 4))


def test_eager_addmm_act_rejects_an_unknown_activation():
    with pytest.raises(ValueError, match="Unexpected activation"):
        eager_addmm_act(torch.nn.SiLU, _Linear(), torch.randn(2, 4))


def test_install_patches_vitdet_once_and_is_idempotent(monkeypatch):
    def _original(activation, linear, mat1):  # pragma: no cover - never called
        raise AssertionError("fused kernel should have been replaced")

    vitdet = types.ModuleType("sam3.model.vitdet")
    vitdet.addmm_act = _original
    sam3 = types.ModuleType("sam3")
    model_pkg = types.ModuleType("sam3.model")
    model_pkg.vitdet = vitdet
    monkeypatch.setitem(sys.modules, "sam3", sam3)
    monkeypatch.setitem(sys.modules, "sam3.model", model_pkg)
    monkeypatch.setitem(sys.modules, "sam3.model.vitdet", vitdet)

    assert install_grad_safe_addmm_act() is True
    assert vitdet.addmm_act is eager_addmm_act
    assert install_grad_safe_addmm_act() is False


def test_install_is_a_no_op_without_sam3(monkeypatch):
    """Importable outside the sidecar env, where sam3 is absent."""
    real_find_spec = sys.meta_path

    class _Blocker:
        def find_spec(self, name, target=None, path=None):
            if name.startswith("sam3"):
                raise ImportError("sam3 blocked")
            return None

    monkeypatch.setattr(sys, "meta_path", [_Blocker()] + list(real_find_spec))
    for mod in [m for m in sys.modules if m.startswith("sam3")]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    assert install_grad_safe_addmm_act() is False


def test_eager_addmm_act_calls_a_wrapper_without_a_weight_attribute():
    """LoraLinear has no `.weight`; unpacking it also skipped the adapter."""
    wrapper = _NoWeightWrapper()
    x = torch.randn(2, 4)
    expected = torch.nn.functional.gelu(wrapper(x))
    torch.testing.assert_close(eager_addmm_act(torch.nn.GELU, wrapper, x), expected)

    eager_addmm_act(torch.nn.GELU, wrapper, x).sum().backward()
    assert wrapper.extra.grad is not None, "adapter branch got no gradient"
