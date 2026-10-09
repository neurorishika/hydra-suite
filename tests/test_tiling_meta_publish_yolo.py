import json
from pathlib import Path

import hydra_suite.training.model_publish as mp
from hydra_suite.core.inference.slice_meta import (
    _training_values,
    sidecar_path,
    write_slice_meta,
)
from hydra_suite.core.inference.tiling_meta import read_tiling_meta
from hydra_suite.training.contracts import TrainingRole

MANIFEST = {
    "geometry_mode": "auto_object",
    "imgsz": 640,
    "object_tile_fraction": 0.1,
    "slice_width": 0,
    "slice_height": 0,
    "overlap": 0.2,
    "min_area_ratio": 0.25,
    "negative_tile_fraction": 0.15,
    "target_sizes": [32, 64, 96, 128],
    "full_frame_mix": True,
    "reference_body_px": 41.5,
    "multiscale_loss_balance": {"enabled": True, "power": 0.5},
}


def _publish_direct_model(
    tmp_path, monkeypatch, *, slice_geometry, source_sidecar=None
):
    monkeypatch.setattr(mp, "get_models_root", lambda: tmp_path)
    src = tmp_path / "weights.pt"
    src.write_bytes(b"fake-weights")
    if source_sidecar is not None:
        write_slice_meta(src, source_sidecar)
    _key, stored = mp.publish_trained_model(
        role=TrainingRole.OBB_DIRECT,
        artifact_path=str(src),
        size="s",
        species="ant",
        model_info="sliced",
        trained_from_run_id="r1",
        dataset_fingerprint="fp",
        base_model="yolo26s-obb.pt",
        slice_geometry=slice_geometry,
    )
    return Path(stored)


def test_publish_writes_additive_v3(tmp_path, monkeypatch):
    dst = _publish_direct_model(tmp_path, monkeypatch, slice_geometry=MANIFEST)
    doc = json.loads(sidecar_path(dst).read_text())
    assert doc["schema_version"] == 3 and doc["model_family"] == "yolo"
    geometry = doc["training_geometry"]
    assert {k: geometry[k] for k in MANIFEST} == MANIFEST
    assert "object_tile_fractions" in geometry
    assert _training_values(geometry) == _training_values(MANIFEST)
    meta = read_tiling_meta(dst)
    assert meta.training.object_tile_fractions == (0.05, 0.1, 0.15, 0.2)
    reg = mp.load_model_registry()
    assert any(
        entry.get("slice_geometry") == MANIFEST for entry in reg["entries"].values()
    )


def test_republish_upgrades_v2_and_keeps_profiles(tmp_path, monkeypatch):
    """Review Focus 5."""
    profile = {
        "id": "bal-1",
        "name": "Balanced",
        "note": "n",
        "settings": {"overlap": 0.3, "object_tile_fraction": 0.12},
        "measurement": {"f1": 0.91, "checkpoint_fingerprint": "sha256:abc"},
    }
    dst = _publish_direct_model(
        tmp_path,
        monkeypatch,
        slice_geometry=MANIFEST,
        source_sidecar={
            "schema_version": 2,
            "training_geometry": MANIFEST,
            "primary_profile_id": "bal-1",
            "profiles": [profile],
        },
    )
    doc = json.loads(sidecar_path(dst).read_text())
    assert doc["schema_version"] == 3
    assert doc["profiles"] == [profile]
    assert doc["primary_profile_id"] == "bal-1"
