"""A published key must resolve, and a mis-loading checkpoint must raise."""

import json

import pytest
import torch

from hydra_suite.core.inference.semantic import checkpoints as ck
from hydra_suite.core.inference.semantic.sam3 import assert_checkpoint_loaded


def test_probe_dependencies_is_variant_independent(monkeypatch):
    # probe_availability rejected anything not in SAM3_VARIANTS, so every
    # published model read as "Unknown SAM3 variant" and stayed disabled.
    monkeypatch.setattr(ck, "_find_spec", lambda n: object())
    # Also stub the predictor-symbol seam: without this the test's outcome
    # depends on whether the box's installed ultralytics happens to expose
    # SAM3SemanticPredictor, making it pass/fail for an environmental reason
    # unrelated to what it's checking (final whole-branch review, M1).
    monkeypatch.setattr(ck, "_has_predictor_symbol", lambda: True)
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


def test_guard_raises_on_an_uncovered_live_key(tmp_path):
    # Coverage runs LIVE -> CHECKPOINT: a live key the checkpoint never
    # carried means those weights stayed stock.
    meta = {"stripped_keys": ["a.weight"], "tuned_fingerprints": {}}
    live = {"a.weight": torch.zeros(2, 2), "b.weight": torch.zeros(2, 2)}
    with pytest.raises(RuntimeError, match="b.weight"):
        assert_checkpoint_loaded(live, meta)


def test_guard_passes_when_checkpoint_is_a_superset(tmp_path):
    # Real, correct shape: the checkpoint carries non-persistent buffers and
    # point-prompt modules the semantic build never instantiates.
    meta = {
        "stripped_keys": [
            "a.weight",
            "blocks.0.attn.freqs_cis",
            "geometry_encoder.points_direct_project.weight",
        ],
        "tuned_fingerprints": {},
    }
    assert_checkpoint_loaded({"a.weight": torch.zeros(2, 2)}, meta)


@pytest.mark.parametrize(
    "live, stripped",
    [({}, ["a.weight"]), ({"a.weight": torch.zeros(2, 2)}, [])],
)
def test_guard_refuses_a_vacuous_coverage_check(live, stripped):
    meta = {"stripped_keys": stripped, "tuned_fingerprints": {}}
    with pytest.raises(RuntimeError, match="vacuous"):
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


def test_guard_against_a_real_published_sidecar(tmp_path):
    """End-to-end regression for the namespace bug (fix round 1, C1).

    All 7 tests above hand-write an idealised `meta` dict. This one derives
    the sidecar from the actual `publish_sam3_model` path, whose
    `tuned_fingerprints` keys came out of `adapter_touched_keys` in the
    PRE-STRIP `detector.<path>.weight` namespace -- a namespace the live,
    already-stripped model state dict never uses. A bare-subscript guard
    KeyErrors on every real published checkpoint; the fixed guard must
    either pass (tuned weights resident, stripped-key lookup) or raise a
    RuntimeError naming the offending key (tuned weights NOT resident) --
    never KeyError either way.
    """

    from hydra_suite.training.contracts import Sam3LoraParams
    from hydra_suite.training.sam3_lora.publish_worker import publish_sam3_artifact

    # bf16 (not float32): both `_tensor_sha256` implementations normalise
    # with `.float()` before hashing. With float32 fixtures that call is a
    # no-op, so a dropped `.float()` on either side of the duplicated
    # implementation would still agree by accident. bf16 fixtures make a
    # dropped `.float()` change the hashed byte length and fail the assert.
    base = {"detector.qkv.weight": torch.randn(4, 4, dtype=torch.bfloat16)}
    torch.save(base, tmp_path / "base.pt")
    torch.save(
        {
            "qkv.lora_A": torch.randn(2, 4, dtype=torch.bfloat16),
            "qkv.lora_B": torch.randn(4, 2, dtype=torch.bfloat16),
        },
        tmp_path / "adapters.pt",
    )
    art, sidecar = publish_sam3_artifact(
        run_id="r1",
        adapters_path=tmp_path / "adapters.pt",
        base_checkpoint=tmp_path / "base.pt",
        build_manifest={"tile_px": 1007, "reference_body_px": 55.4},
        params=Sam3LoraParams(prompt="ant", rank=2, alpha=4),
        source_fingerprint="fp1",
        models_root=tmp_path / "models",
    )
    meta = json.loads(sidecar.read_text())

    merged = torch.load(art, map_location="cpu", weights_only=True)
    # The live model's state dict is what ultralytics hands back AFTER its
    # substring-strip load transform -- i.e. stripped, not `merged` as-is.
    stripped_live = {
        k.replace("detector.", ""): v for k, v in merged.items() if "detector" in k
    }

    # Good path: tuned weights are resident under the stripped namespace.
    assert assert_checkpoint_loaded(stripped_live, meta, imgsz=meta["imgsz"]) is None

    # Bad path: tuned weights are NOT resident (reverted to a base tensor
    # for every tuned key) -- must raise RuntimeError naming a real key,
    # never KeyError.
    tampered = dict(stripped_live)
    for key in meta["tuned_fingerprints"]:
        tampered[key] = torch.zeros_like(tampered[key])
    with pytest.raises(RuntimeError):
        assert_checkpoint_loaded(tampered, meta, imgsz=meta["imgsz"])
