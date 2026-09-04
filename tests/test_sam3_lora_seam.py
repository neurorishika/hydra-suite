"""LoRA inject/merge round-trip, provable without SAM3 or a GPU."""

import pytest
import torch
from torch import nn

from hydra_suite.training.sam3_lora.lora import (
    LoraConfig,
    adapter_state_dict,
    inject_adapters,
    merge_adapters,
)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(8, 8, bias=False)
        self.other = nn.Linear(8, 8, bias=False)


def _cfg(rank=2, alpha=4, **kw):
    return LoraConfig(
        rank=rank, alpha=alpha, dropout=0.0, target_suffixes=("qkv",), **kw
    )


def test_inject_only_wraps_targeted_suffixes():
    m = Toy()
    assert inject_adapters(m, _cfg()) == 1
    assert "qkv" in " ".join(adapter_state_dict(m).keys())
    assert "other" not in " ".join(adapter_state_dict(m).keys())


def test_zero_initialised_adapter_merges_as_a_no_op():
    m = Toy()
    inject_adapters(m, _cfg())
    base = {"detector.qkv.weight": torch.randn(8, 8)}
    merged = merge_adapters(base, adapter_state_dict(m), _cfg())
    # lora_B is zero-initialised, so an untrained adapter must change nothing.
    assert torch.equal(merged["detector.qkv.weight"], base["detector.qkv.weight"])


def test_merge_applies_the_scaled_low_rank_delta():
    m = Toy()
    cfg = _cfg(rank=2, alpha=4)
    inject_adapters(m, cfg)
    sd = adapter_state_dict(m)
    a_key = next(k for k in sd if k.endswith("lora_A"))
    b_key = next(k for k in sd if k.endswith("lora_B"))
    sd[b_key] = torch.randn_like(sd[b_key])
    w = torch.randn(8, 8)
    merged = merge_adapters({"detector.qkv.weight": w}, sd, cfg)
    expected = w + (sd[b_key] @ sd[a_key]) * (cfg.alpha / cfg.rank)
    assert torch.allclose(merged["detector.qkv.weight"], expected, atol=1e-6)


def test_unresolved_adapter_is_a_hard_error():
    m = Toy()
    inject_adapters(m, _cfg())
    # A silent skip is indistinguishable from a successful merge and yields a
    # checkpoint that differs from base in bytes but not in behaviour.
    with pytest.raises(KeyError):
        merge_adapters(
            {"detector.somethingelse.weight": torch.randn(8, 8)},
            adapter_state_dict(m),
            _cfg(),
        )


class Nested(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = Toy()
        self.text = Toy()


def test_include_prefixes_scope_injection():
    m = Nested()
    # The six adapt_* flags select submodules by PREFIX; without this the
    # flags are dead parameters and the frozen text encoder gets adapted.
    n = inject_adapters(m, _cfg(include_prefixes=("vision",)))
    assert n == 1
    keys = " ".join(adapter_state_dict(m).keys())
    assert "vision" in keys and "text" not in keys


def test_exclude_prefixes_win_over_include():
    m = Nested()
    n = inject_adapters(
        m, _cfg(include_prefixes=("vision", "text"), exclude_prefixes=("text",))
    )
    assert n == 1
    assert "text" not in " ".join(adapter_state_dict(m).keys())


def test_empty_include_prefixes_means_everything():
    m = Nested()
    assert inject_adapters(m, _cfg()) == 2


def test_every_submodule_prefix_is_nonempty():
    from hydra_suite.training.sam3_lora.lora import SUBMODULE_PREFIXES

    # A prefix matching no module silently disables its flag. These values were
    # MEASURED against a live model; this pins the shape at least.
    assert set(SUBMODULE_PREFIXES) == {
        "adapt_vision_encoder",
        "adapt_text_encoder",
        "adapt_geometry_encoder",
        "adapt_detr_encoder",
        "adapt_detr_decoder",
        "adapt_mask_decoder",
    }
    assert all(
        v and all(isinstance(x, str) and x for x in v)
        for v in SUBMODULE_PREFIXES.values()
    )


def test_lora_config_from_params_maps_the_flags():
    from types import SimpleNamespace

    from hydra_suite.training.sam3_lora.lora import (
        SUBMODULE_PREFIXES,
        lora_config_from_params,
    )

    # Duck-typed, NOT Sam3LoraParams: that lands in Task 5, which runs AFTER
    # this task. lora.py must not import contracts either -- it is a pure
    # tensor seam with no SAM3 and no contracts dependency.
    params = SimpleNamespace(
        rank=8,
        alpha=16,
        dropout=0.0,
        adapt_vision_encoder=True,
        adapt_text_encoder=False,
        adapt_geometry_encoder=False,
        adapt_detr_encoder=False,
        adapt_detr_decoder=False,
        adapt_mask_decoder=False,
    )
    cfg = lora_config_from_params(params)
    for pref in SUBMODULE_PREFIXES["adapt_text_encoder"]:
        assert pref not in cfg.include_prefixes
    for pref in SUBMODULE_PREFIXES["adapt_vision_encoder"]:
        assert pref in cfg.include_prefixes


def test_lora_config_all_scopes_uses_explicit_budgeted_prefixes():
    from types import SimpleNamespace

    from hydra_suite.training.sam3_lora.lora import (
        SUBMODULE_PREFIXES,
        lora_config_from_params,
    )

    params = SimpleNamespace(
        rank=8,
        alpha=16,
        dropout=0.0,
        **{flag: True for flag in SUBMODULE_PREFIXES},
    )
    config = lora_config_from_params(params)

    assert set(config.include_prefixes) == {
        prefix for prefixes in SUBMODULE_PREFIXES.values() for prefix in prefixes
    }
    assert "dot_prod_scoring" not in config.include_prefixes


def test_lora_config_rejects_no_enabled_scope():
    from types import SimpleNamespace

    from hydra_suite.training.sam3_lora.lora import (
        SUBMODULE_PREFIXES,
        lora_config_from_params,
    )

    params = SimpleNamespace(
        rank=8,
        alpha=16,
        dropout=0.0,
        **{flag: False for flag in SUBMODULE_PREFIXES},
    )

    with pytest.raises(ValueError, match="scope"):
        lora_config_from_params(params)


def test_non_targeted_base_keys_pass_through_untouched():
    m = Toy()
    inject_adapters(m, _cfg())
    base = {"detector.qkv.weight": torch.randn(8, 8), "detector.buffer": torch.randn(3)}
    merged = merge_adapters(base, adapter_state_dict(m), _cfg())
    assert torch.equal(merged["detector.buffer"], base["detector.buffer"])
