"""The merged artifact must load through ultralytics' own key transform."""

import json

import torch

from hydra_suite.training.contracts import Sam3LoraParams
from hydra_suite.training.sam3_lora.publish import publish_sam3_model, stripped_keys


def test_stripped_keys_reproduce_ultralytics_transform():
    sd = {
        "detector.a.weight": torch.zeros(1),
        "other.b.weight": torch.zeros(1),
        "x.detector.c": torch.zeros(1),
    }
    # build_sam3.py:357 filters on the SUBSTRING "detector", not a prefix.
    got = set(stripped_keys(sd))
    assert "a.weight" in got
    assert "other.b.weight" not in got
    assert "x.c" in got


def test_artifact_is_not_written_into_the_stock_cache(tmp_path):
    base = {"detector.qkv.weight": torch.randn(4, 4)}
    torch.save(base, tmp_path / "base.pt")
    torch.save(
        {"qkv.lora_A": torch.zeros(2, 4), "qkv.lora_B": torch.zeros(4, 2)},
        tmp_path / "adapters.pt",
    )
    key, art = publish_sam3_model(
        run_id="r1",
        adapters_path=tmp_path / "adapters.pt",
        base_checkpoint=tmp_path / "base.pt",
        build_manifest={"tile_px": 1007, "reference_body_px": 55.4},
        params=Sam3LoraParams(prompt="ant"),
        source_fingerprint="fp1",
        models_root=tmp_path / "models",
    )
    # get_models_dir()/"sam3" is checkpoints.py's DOWNLOAD CACHE.
    assert "sam3_finetuned" in str(art)
    assert "/sam3/" not in str(art)
    # (key, artifact) -- NOT (artifact, sidecar); run_role_training unpacks
    # this as (published_key, published_path).
    assert key and str(art).endswith(".pt")


def test_sidecar_records_the_guard_fields(tmp_path):
    base = {"detector.qkv.weight": torch.randn(4, 4)}
    torch.save(base, tmp_path / "base.pt")
    torch.save(
        {"qkv.lora_A": torch.randn(2, 4), "qkv.lora_B": torch.randn(4, 2)},
        tmp_path / "adapters.pt",
    )
    _, art = publish_sam3_model(
        run_id="r1",
        adapters_path=tmp_path / "adapters.pt",
        base_checkpoint=tmp_path / "base.pt",
        build_manifest={"tile_px": 1007, "reference_body_px": 55.4},
        params=Sam3LoraParams(prompt="ant"),
        source_fingerprint="fp1",
        models_root=tmp_path / "models",
    )
    # The registry must land under models_root, never the user's real one.
    assert (tmp_path / "models" / "model_registry.json").exists()
    side = str(art) + ".sam3_meta.json"
    meta = json.loads(open(side).read())
    assert meta["prompt"] == "ant"
    assert meta["train_tile_px"] == 1007
    assert meta["imgsz"] == 1008
    assert meta["stripped_keys"]
    assert meta["tuned_fingerprints"]  # sha256 of tensors the merge changed
    assert meta["source_fingerprint"] == "fp1"
    assert meta["reference_body_px"] == 55.4
