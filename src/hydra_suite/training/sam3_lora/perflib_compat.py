"""Make SAM3's vision trunk differentiable.

``sam3/model/vitdet.py``'s ``Mlp.forward`` calls ``perflib.fused.addmm_act``
unconditionally. That kernel is inference-only in two separate ways:

* it raises ``ValueError("Expected grad to be disabled.")`` whenever
  ``torch.is_grad_enabled()``, and
* it ``.detach()``es both the weight and the bias, so even with grad enabled
  no gradient could ever flow back into ``fc1``.

There is no ``USE_PERFLIB`` escape on this path -- ``perflib.is_enabled``
gates other call sites, but ``Mlp.forward`` imports ``addmm_act`` directly --
and the released ``sam3/train/`` package never references perflib at all, so
upstream ships no trainable variant of this module.

We therefore rebind the module-global to an eager, autograd-friendly
equivalent. ``Mlp.forward`` resolves ``addmm_act`` at call time, so patching
the module attribute is enough and no vendored file is edited. The fused
kernel's bf16 casts are deliberately NOT reproduced: the sidecar already runs
under bf16 autocast, which handles precision at the right level, and hard
casts here would silently downcast an fp32 run.
"""

from __future__ import annotations

from typing import Any


def eager_addmm_act(activation: Any, linear: Any, mat1: Any) -> Any:
    """``activation(mat1 @ linear.weight.T + linear.bias)``, differentiable.

    Mirrors ``perflib.fused.addmm_act``'s contract -- *activation* is the
    activation CLASS (``type(self.act)``), not an instance -- but keeps the
    weight and bias attached to the graph.
    """
    import torch
    import torch.nn.functional as F

    y = F.linear(mat1, linear.weight, linear.bias)
    if activation in (F.relu, torch.nn.ReLU):
        return F.relu(y)
    if activation in (F.gelu, torch.nn.GELU):
        return F.gelu(y)
    raise ValueError(f"Unexpected activation {activation}")


def install_grad_safe_addmm_act() -> bool:
    """Rebind ``vitdet.addmm_act``. Returns True if the patch was applied.

    Idempotent: a second call is a no-op. Raises nothing when ``sam3`` is
    absent -- callers outside the sidecar env must stay importable.
    """
    try:
        from sam3.model import vitdet
    except ImportError:
        return False
    if getattr(vitdet.addmm_act, "_hydra_grad_safe", False):
        return False
    eager_addmm_act._hydra_grad_safe = True  # type: ignore[attr-defined]
    vitdet.addmm_act = eager_addmm_act
    return True
