"""Atomic, consumer-compatible SAM3 checkpoint publication."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from hydra_suite.core.inference.semantic.sam3 import assert_checkpoint_loaded
from hydra_suite.training.contracts import Sam3LoraParams
from hydra_suite.training.sam3_lora import publish_worker


def _inputs(tmp_path: Path, *, dtype=torch.float32):
    tuned = torch.randn(4, 4, dtype=dtype)
    stock = torch.randn(3, 3, dtype=dtype)
    base = {
        "detector.qkv.weight": tuned,
        "detector.vision_backbone.sam2_convs.0.weight": stock,
        "stock.metadata_tensor": torch.arange(3),
    }
    base_path = tmp_path / "base.pt"
    # The trusted Meta source may be wrapped, but the existing published
    # consumer contract is the unwrapped detector-prefixed state dict.
    torch.save({"model": base, "epoch": 12}, base_path)
    adapters = {
        "qkv.lora_A": torch.randn(2, 4, dtype=dtype),
        "qkv.lora_B": torch.randn(4, 2, dtype=dtype),
    }
    adapters_path = tmp_path / "adapters.pt"
    torch.save(adapters, adapters_path)
    return base, adapters, base_path, adapters_path


def _publish(tmp_path: Path, **kwargs):
    base, adapters, base_path, adapters_path = _inputs(
        tmp_path, dtype=kwargs.pop("dtype", torch.float32)
    )
    artifact, sidecar = publish_worker.publish_sam3_artifact(
        run_id="run-1",
        adapters_path=adapters_path,
        base_checkpoint=base_path,
        build_manifest={
            "tile_px": 1007,
            "reference_body_px": 55.4,
            "object_tile_fraction": 0.055,
        },
        params=Sam3LoraParams(
            prompt="ant", rank=2, alpha=4, label_quality_acknowledged=True
        ),
        source_fingerprint="fp1",
        models_root=tmp_path / "models",
        **kwargs,
    )
    return base, adapters, artifact, sidecar


def _staging_paths(models_root: Path) -> list[Path]:
    out_dir = models_root / "sam3_finetuned"
    return list(out_dir.glob(".*.tmp")) if out_dir.exists() else []


def test_atomic_publish_preserves_layout_dtype_stock_keys_and_numerics(tmp_path):
    base, adapters, artifact, sidecar_path = _publish(tmp_path, dtype=torch.bfloat16)

    merged = torch.load(artifact, map_location="cpu", weights_only=True)
    expected = base["detector.qkv.weight"].clone()
    expected.add_((adapters["qkv.lora_B"] @ adapters["qkv.lora_A"]) * 2.0)
    assert set(merged) == set(base)
    assert "model" not in merged
    assert merged["detector.qkv.weight"].dtype == torch.bfloat16
    assert torch.equal(merged["detector.qkv.weight"], expected)
    assert torch.equal(
        merged["detector.vision_backbone.sam2_convs.0.weight"],
        base["detector.vision_backbone.sam2_convs.0.weight"],
    )
    assert torch.equal(merged["stock.metadata_tensor"], base["stock.metadata_tensor"])

    meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
    stripped_live = {
        key.replace("detector.", ""): value
        for key, value in merged.items()
        if "detector" in key
    }
    assert_checkpoint_loaded(stripped_live, meta, imgsz=meta["imgsz"])
    assert meta["prompt"] == "ant"
    assert meta["train_tile_px"] == 1007
    assert meta["source_fingerprint"] == "fp1"
    assert meta["label_quality_acknowledged"] is True
    assert meta["stripped_keys"] == [
        "qkv.weight",
        "vision_backbone.sam2_convs.0.weight",
    ]


def test_publish_preserves_first_three_touched_fingerprint_contract(tmp_path):
    base = {
        "detector.a.weight": torch.randn(4, 4),
        "detector.b.weight": torch.randn(4, 4),
    }
    base_path = tmp_path / "base.pt"
    torch.save(base, base_path)
    adapters_path = tmp_path / "adapters.pt"
    torch.save(
        {
            "a.lora_A": torch.ones(2, 4),
            "a.lora_B": torch.ones(4, 2),
            "b.lora_A": torch.ones(2, 4),
            # Existing metadata records each of the first three touched keys,
            # including a no-op key when another recorded adapter did change.
            "b.lora_B": torch.zeros(4, 2),
        },
        adapters_path,
    )

    _artifact, sidecar = publish_worker.publish_sam3_artifact(
        run_id="run-1",
        adapters_path=adapters_path,
        base_checkpoint=base_path,
        build_manifest={},
        params=Sam3LoraParams(prompt="ant", rank=2, alpha=4),
        source_fingerprint="fp1",
        models_root=tmp_path / "models",
    )

    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert list(metadata["tuned_fingerprints"]) == ["a.weight", "b.weight"]


def test_missing_mapping_fails_before_any_artifact_or_staging_is_visible(tmp_path):
    _base, _adapters, base_path, adapters_path = _inputs(tmp_path)
    torch.save(
        {
            "qkv.lora_A": torch.randn(2, 4),
            "qkv.lora_B": torch.randn(4, 2),
            "z_missing.lora_A": torch.randn(2, 4),
            "z_missing.lora_B": torch.randn(4, 2),
        },
        adapters_path,
    )

    with pytest.raises(KeyError, match="z_missing"):
        publish_worker.publish_sam3_artifact(
            run_id="run-1",
            adapters_path=adapters_path,
            base_checkpoint=base_path,
            build_manifest={},
            params=Sam3LoraParams(prompt="ant", rank=2, alpha=4),
            source_fingerprint="fp1",
            models_root=tmp_path / "models",
        )

    assert not (tmp_path / "models" / "sam3_finetuned" / "run-1.pt").exists()
    assert _staging_paths(tmp_path / "models") == []


@pytest.mark.parametrize("failure_site", ["save", "validation"])
def test_save_or_validation_failure_leaves_no_final_or_staging(
    monkeypatch, tmp_path, failure_site
):
    _base, _adapters, base_path, adapters_path = _inputs(tmp_path)
    target = (
        "_save_checkpoint" if failure_site == "save" else "_validate_staged_artifact"
    )
    monkeypatch.setattr(
        publish_worker,
        target,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(failure_site)),
    )

    with pytest.raises(RuntimeError, match=failure_site):
        publish_worker.publish_sam3_artifact(
            run_id="run-1",
            adapters_path=adapters_path,
            base_checkpoint=base_path,
            build_manifest={},
            params=Sam3LoraParams(prompt="ant", rank=2, alpha=4),
            source_fingerprint="fp1",
            models_root=tmp_path / "models",
        )

    out_dir = tmp_path / "models" / "sam3_finetuned"
    assert not (out_dir / "run-1.pt").exists()
    assert not (out_dir / "run-1.pt.sam3_meta.json").exists()
    assert _staging_paths(tmp_path / "models") == []


def test_second_atomic_promotion_failure_rolls_back_the_first(monkeypatch, tmp_path):
    _base, _adapters, base_path, adapters_path = _inputs(tmp_path)
    real_replace = publish_worker._atomic_replace
    calls = 0

    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("promotion")
        return real_replace(source, target)

    monkeypatch.setattr(publish_worker, "_atomic_replace", fail_second)
    with pytest.raises(OSError, match="promotion"):
        publish_worker.publish_sam3_artifact(
            run_id="run-1",
            adapters_path=adapters_path,
            base_checkpoint=base_path,
            build_manifest={},
            params=Sam3LoraParams(prompt="ant", rank=2, alpha=4),
            source_fingerprint="fp1",
            models_root=tmp_path / "models",
        )

    out_dir = tmp_path / "models" / "sam3_finetuned"
    assert not (out_dir / "run-1.pt").exists()
    assert not (out_dir / "run-1.pt.sam3_meta.json").exists()
    assert _staging_paths(tmp_path / "models") == []


def test_post_promotion_fsync_failure_removes_both_final_names(monkeypatch, tmp_path):
    _base, _adapters, base_path, adapters_path = _inputs(tmp_path)
    real_fsync = publish_worker._fsync_directory
    calls = 0

    def fail_once(directory):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("directory fsync")
        return real_fsync(directory)

    monkeypatch.setattr(publish_worker, "_fsync_directory", fail_once)
    with pytest.raises(OSError, match="directory fsync"):
        publish_worker.publish_sam3_artifact(
            run_id="run-1",
            adapters_path=adapters_path,
            base_checkpoint=base_path,
            build_manifest={},
            params=Sam3LoraParams(prompt="ant", rank=2, alpha=4),
            source_fingerprint="fp1",
            models_root=tmp_path / "models",
        )

    out_dir = tmp_path / "models" / "sam3_finetuned"
    assert not (out_dir / "run-1.pt").exists()
    assert not (out_dir / "run-1.pt.sam3_meta.json").exists()
    assert _staging_paths(tmp_path / "models") == []


def test_existing_published_pair_is_never_overwritten(tmp_path):
    _base, _adapters, base_path, adapters_path = _inputs(tmp_path)
    out_dir = tmp_path / "models" / "sam3_finetuned"
    out_dir.mkdir(parents=True)
    artifact = out_dir / "run-1.pt"
    sidecar = out_dir / "run-1.pt.sam3_meta.json"
    artifact.write_bytes(b"old-checkpoint")
    sidecar.write_bytes(b"old-sidecar")

    with pytest.raises(FileExistsError):
        publish_worker.publish_sam3_artifact(
            run_id="run-1",
            adapters_path=adapters_path,
            base_checkpoint=base_path,
            build_manifest={},
            params=Sam3LoraParams(prompt="ant", rank=2, alpha=4),
            source_fingerprint="fp1",
            models_root=tmp_path / "models",
        )

    assert artifact.read_bytes() == b"old-checkpoint"
    assert sidecar.read_bytes() == b"old-sidecar"
    assert _staging_paths(tmp_path / "models") == []


def test_raced_publish_target_is_not_overwritten_during_promotion(
    monkeypatch, tmp_path
):
    _base, _adapters, base_path, adapters_path = _inputs(tmp_path)
    real_validate = publish_worker._validate_staged_artifact
    out_dir = tmp_path / "models" / "sam3_finetuned"
    artifact = out_dir / "run-1.pt"
    sidecar = out_dir / "run-1.pt.sam3_meta.json"

    def validate_then_race(*args, **kwargs):
        real_validate(*args, **kwargs)
        artifact.write_bytes(b"concurrent-checkpoint")
        sidecar.write_bytes(b"concurrent-sidecar")

    monkeypatch.setattr(publish_worker, "_validate_staged_artifact", validate_then_race)
    with pytest.raises(FileExistsError):
        publish_worker.publish_sam3_artifact(
            run_id="run-1",
            adapters_path=adapters_path,
            base_checkpoint=base_path,
            build_manifest={},
            params=Sam3LoraParams(prompt="ant", rank=2, alpha=4),
            source_fingerprint="fp1",
            models_root=tmp_path / "models",
        )

    assert artifact.read_bytes() == b"concurrent-checkpoint"
    assert sidecar.read_bytes() == b"concurrent-sidecar"
    assert _staging_paths(tmp_path / "models") == []
