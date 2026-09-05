"""Per-module adapter scopes: the `dot_prod_scoring` head (spike parity).

The audit (`docs/superpowers/specs/2026-09-05-sam3-finetune-quality-audit.md`,
N1) found `dot_prod_scoring.prompt_proj` is the research spike's single
biggest mover (lora_B norm 0.174) among modules the spike adapts and we do
not.  These tests pin the surface exactly: two Linears, no more.

The stub trees below mirror the real dotted paths verbatim so the whole scope
mechanism is provable without `import sam3` (unavailable on macOS: triton).
"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hydra_suite.training.sam3_lora.lora import (
    SUBMODULE_PATHS,
    SUBMODULE_PREFIXES,
    LoraLinear,
    adapter_state_dict,
    adapter_touched_keys,
    inject_adapters,
    lora_config_from_params,
    merge_adapters,
)


class _DotProdScoring(nn.Module):
    """The scoring head: 4 Linears, of which the spike trained exactly 2.

    `proj` is a deliberate decoy -- it is a literal `TARGET_SUFFIXES` entry, so
    a prefix-based scope would wrap it and silently exceed the spike surface.
    """

    def __init__(self) -> None:
        super().__init__()
        self.prompt_proj = nn.Linear(8, 8, bias=False)
        self.hs_proj = nn.Linear(8, 8, bias=False)
        self.proj = nn.Linear(8, 8, bias=False)
        self.geometry_proj = nn.Linear(8, 8, bias=False)


class _Sam3Stub(nn.Module):
    """Mirrors the real module paths: scoring head, its deepcopy, one decoy."""

    def __init__(self) -> None:
        super().__init__()
        self.dot_prod_scoring = _DotProdScoring()
        # sam3_image.py:83-85 -- `instance_dot_prod_scoring = deepcopy(...)`.
        self.instance_dot_prod_scoring = deepcopy(self.dot_prod_scoring)
        self.backbone = nn.Module()
        self.backbone.vision_backbone = nn.Module()
        self.backbone.vision_backbone.qkv = nn.Linear(8, 8, bias=False)


def _params(**flags):
    base = {flag: False for flag in SUBMODULE_PREFIXES}
    base.update({flag: False for flag in SUBMODULE_PATHS})
    base.update(flags)
    return SimpleNamespace(rank=2, alpha=4, dropout=0.0, **base)


def _wrapped(model: nn.Module) -> set[str]:
    return {n for n, m in model.named_modules() if isinstance(m, LoraLinear)}


def test_scoring_head_flag_wraps_exactly_prompt_proj_and_hs_proj():
    model = _Sam3Stub()
    cfg = lora_config_from_params(_params(adapt_scoring_head=True))
    count = inject_adapters(model, cfg)

    assert _wrapped(model) == {
        "dot_prod_scoring.prompt_proj",
        "dot_prod_scoring.hs_proj",
    }
    assert count == 2


def test_scoring_head_excludes_the_instance_scoring_deepcopy():
    """`instance_dot_prod_scoring` exclusion IS the intended spike parity.

    The spike checkpoint carries no adapters for it; wrapping it would exceed
    the validated surface.
    """
    model = _Sam3Stub()
    inject_adapters(model, lora_config_from_params(_params(adapt_scoring_head=True)))

    assert not any(
        name.startswith("instance_dot_prod_scoring") for name in _wrapped(model)
    )


def test_scoring_head_flag_off_wraps_no_scoring_module():
    model = _Sam3Stub()
    cfg = lora_config_from_params(_params(adapt_vision_encoder=True))
    inject_adapters(model, cfg)

    assert _wrapped(model) == {"backbone.vision_backbone.qkv"}
    assert not cfg.include_module_paths


def test_scoring_only_scope_does_not_fall_back_to_matching_everything():
    """The empty-`include_prefixes` sentinel must not mean "the whole model"."""
    model = _Sam3Stub()
    inject_adapters(model, lora_config_from_params(_params(adapt_scoring_head=True)))

    assert "backbone.vision_backbone.qkv" not in _wrapped(model)


def test_scoring_head_is_a_scope_for_the_no_scope_refusal():
    cfg = lora_config_from_params(_params(adapt_scoring_head=True))
    assert cfg.include_module_paths

    with pytest.raises(ValueError, match="scope"):
        lora_config_from_params(_params())


def test_scoring_head_adapters_merge_onto_their_own_base_weights():
    model = _Sam3Stub()
    cfg = lora_config_from_params(_params(adapt_scoring_head=True))
    inject_adapters(model, cfg)
    adapters = adapter_state_dict(model)
    for key in adapters:
        if key.endswith("lora_B"):
            adapters[key] = torch.randn_like(adapters[key])

    base = {
        "detector.dot_prod_scoring.prompt_proj.weight": torch.zeros(8, 8),
        "detector.dot_prod_scoring.hs_proj.weight": torch.zeros(8, 8),
        "detector.dot_prod_scoring.proj.weight": torch.zeros(8, 8),
    }
    touched = adapter_touched_keys(adapters, base)
    assert touched == {
        "detector.dot_prod_scoring.prompt_proj.weight",
        "detector.dot_prod_scoring.hs_proj.weight",
    }

    merged = merge_adapters(base, adapters, cfg)
    expected = (
        adapters["dot_prod_scoring.prompt_proj.lora_B"]
        @ adapters["dot_prod_scoring.prompt_proj.lora_A"]
    ) * cfg.scaling
    assert torch.allclose(
        merged["detector.dot_prod_scoring.prompt_proj.weight"], expected
    )
    assert torch.equal(
        merged["detector.dot_prod_scoring.proj.weight"], torch.zeros(8, 8)
    )
