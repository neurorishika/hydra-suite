"""A published key must resolve, and a mis-loading checkpoint must raise."""

import pytest
import torch

from hydra_suite.core.inference.semantic import checkpoints as ck
from hydra_suite.core.inference.semantic.sam3 import assert_checkpoint_loaded


def test_probe_dependencies_is_variant_independent(monkeypatch):
    # probe_availability rejected anything not in SAM3_VARIANTS, so every
    # published model read as "Unknown SAM3 variant" and stayed disabled.
    monkeypatch.setattr(ck, "_find_spec", lambda n: object())
    assert ck.probe_dependencies().usable


def test_available_models_includes_registry_entries(monkeypatch):
    monkeypatch.setattr(ck, "_registry_semantic_models", lambda: ["run123"])
    got = ck.available_models()
    assert "sam3" in got and "run123" in got


def test_guard_raises_when_tuned_tensors_are_absent(tmp_path):
    # The failure this guard exists for: all keys present, but the model holds
    # BASE weights because ultralytics' load-time transform changed.
    meta = {
        "stripped_keys": ["a.weight"],
        "tuned_fingerprints": {"a.weight": "deadbeef"},
    }
    live = {"a.weight": torch.zeros(2, 2)}
    with pytest.raises(RuntimeError, match="a.weight"):
        assert_checkpoint_loaded(live, meta)


def test_guard_raises_on_a_missing_key(tmp_path):
    meta = {"stripped_keys": ["a.weight", "b.weight"], "tuned_fingerprints": {}}
    live = {"a.weight": torch.zeros(2, 2)}
    with pytest.raises(RuntimeError, match="b.weight"):
        assert_checkpoint_loaded(live, meta)


def test_guard_passes_when_fingerprints_match():
    import hashlib

    t = torch.randn(2, 2)
    fp = hashlib.sha256(t.numpy().tobytes()).hexdigest()
    meta = {
        "stripped_keys": ["a.weight"],
        "tuned_fingerprints": {"a.weight": fp},
        "imgsz": 1008,
    }
    assert assert_checkpoint_loaded({"a.weight": t}, meta, imgsz=1008) is None


def test_guard_refuses_an_imgsz_mismatch():
    import hashlib

    # A model finetuned at 1008 and served at 644 is a 1.56x train/serve scale
    # mismatch. It loads CLEANLY -- keys and tensors all match -- so only an
    # explicit check catches it. Rescaling silently is the failure mode.
    t = torch.randn(2, 2)
    fp = hashlib.sha256(t.numpy().tobytes()).hexdigest()
    meta = {
        "stripped_keys": ["a.weight"],
        "tuned_fingerprints": {"a.weight": fp},
        "imgsz": 1008,
    }
    with pytest.raises(RuntimeError, match="644"):
        assert_checkpoint_loaded({"a.weight": t}, meta, imgsz=644)


def test_stock_variant_without_a_sidecar_is_unguarded():
    # A stock variant ships no sidecar and makes no claim; guarding it would
    # refuse every un-finetuned run.
    assert (
        assert_checkpoint_loaded({"a.weight": torch.zeros(2, 2)}, None, imgsz=1008)
        is None
    )
