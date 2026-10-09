import json
import logging
from pathlib import Path

import pytest

from hydra_suite.core.inference.slice_meta import sidecar_path, write_slice_meta
from hydra_suite.core.inference.tiling_meta import read_tiling_meta
from hydra_suite.training.contracts import Sam3LoraParams
from hydra_suite.training.sam3_lora import publish, publish_worker

FULL_MANIFEST = {
    "type": "sam3_coco_tiles",
    "geometry_mode": "auto_object",
    "tile_px_set": [[1940, 1940], [970, 970]],
    "object_tile_fractions": [0.0275, 0.055],
    "prefill_object_tile_fraction": 0.04125,
    "prefill_tile_px": [970, 970],
    "reference_body_px": 53.4,
    "tile_overlap": 0.25,
    "min_retained_area_frac": 0.3,
    "full_frame_mix": False,
    "scale_range_px": [970, 1940],
}


def _request_kwargs(full):
    """Minimal valid call, as tests/test_sam3_publish.py builds it."""
    return {
        "run_id": "r",
        "adapters_path": Path("/tmp/a.pt"),
        "base_checkpoint": Path("/tmp/b.pt"),
        "build_manifest": full,
        "params": Sam3LoraParams(prompt="ant", label_quality_acknowledged=True),
        "source_fingerprint": "fp",
        "models_root": Path("/tmp/models"),
        "attempt_id": "a",
    }


def _child_manifest(full):
    """What the publish child really receives (adversarial M1)."""
    payload = publish._request_payload(**_request_kwargs(full))
    return payload["build_manifest"]


def test_child_manifest_carries_tiling_fields():
    child = _child_manifest(FULL_MANIFEST)
    assert child["geometry_mode"] == "auto_object"
    assert child["tile_overlap"] == 0.25
    assert child["min_retained_area_frac"] == 0.3


@pytest.mark.parametrize(
    "field,bad",
    [
        ("geometry_mode", "weird"),
        ("tile_overlap", 1.0),
        ("tile_overlap", float("nan")),
        ("min_retained_area_frac", 1.5),
        ("tile_overlap", True),
    ],
)
def test_child_manifest_rejects_invalid_tiling_fields(field, bad):
    with pytest.raises(ValueError):
        _child_manifest(dict(FULL_MANIFEST, **{field: bad}))


def test_tiling_sidecar_from_child_manifest(tmp_path):
    artifact = tmp_path / "sam3-run.pt"
    artifact.write_bytes(b"x")
    written = publish_worker._write_tiling_sidecar(
        artifact, _child_manifest(FULL_MANIFEST)
    )
    assert written == sidecar_path(artifact)
    meta = read_tiling_meta(artifact)
    assert meta.model_family == "sam3" and meta.source == "slice_meta"
    assert meta.training.geometry_mode == "auto_object"
    assert meta.training.overlap == 0.25
    assert meta.training.min_area_ratio == 0.3
    assert meta.tile_px_set == ((1940, 1940), (970, 970))
    assert meta.imgsz == publish_worker.PREDICTOR_IMGSZ


def test_tiling_sidecar_keeps_existing_profiles(tmp_path):
    """Adversarial m3."""
    artifact = tmp_path / "sam3-run.pt"
    artifact.write_bytes(b"x")
    profile = {
        "id": "p-1",
        "name": "Calibrated",
        "note": "",
        "settings": {"object_tile_fraction": 0.05},
        "measurement": {},
    }
    write_slice_meta(
        artifact,
        {
            "schema_version": 3,
            "model_family": "sam3",
            "training_geometry": {},
            "primary_profile_id": "p-1",
            "profiles": [profile],
        },
    )
    publish_worker._write_tiling_sidecar(artifact, _child_manifest(FULL_MANIFEST))
    meta = read_tiling_meta(artifact)
    assert meta.profiles == (profile,) and meta.primary_profile_id == "p-1"


def test_write_failure_is_non_fatal(tmp_path, caplog, monkeypatch):
    """Review Focus 4."""
    caplog.set_level(logging.WARNING)
    artifact = tmp_path / "sam3-run.pt"
    artifact.write_bytes(b"x")

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(publish_worker, "write_slice_meta", boom)
    assert publish_worker._write_tiling_sidecar(artifact, FULL_MANIFEST) is None
    assert "tiling sidecar" in caplog.text


def test_cleanup_removes_owned_tiling_sidecar(tmp_path):
    artifact = tmp_path / "sam3-run.pt"
    sam3_sidecar = tmp_path / "sam3-run.pt.sam3_meta.json"
    artifact.write_bytes(b"x")
    sam3_sidecar.write_text(json.dumps({"publish_attempt_id": "a" * 32}))
    tiling = sidecar_path(artifact)
    tiling.write_text("{}")
    tiling.with_name(tiling.name + ".tmp").write_text("{}")
    publish._cleanup_attempt(
        artifact_path=artifact,
        sidecar_path=sam3_sidecar,
        control_dir=None,
        attempt_id="a" * 32,
    )
    assert not artifact.exists() and not sam3_sidecar.exists()
    assert not tiling.exists() and not tiling.with_name(tiling.name + ".tmp").exists()


def test_cleanup_keeps_unowned_tiling_sidecar(tmp_path):
    artifact = tmp_path / "sam3-run.pt"
    sam3_sidecar = tmp_path / "sam3-run.pt.sam3_meta.json"
    artifact.write_bytes(b"x")
    sam3_sidecar.write_text(json.dumps({"publish_attempt_id": "b" * 32}))
    tiling = sidecar_path(artifact)
    tiling.write_text("{}")
    publish._cleanup_attempt(
        artifact_path=artifact,
        sidecar_path=sam3_sidecar,
        control_dir=None,
        attempt_id="a" * 32,
    )
    assert tiling.exists()


def _owned_pair(tmp_path, attempt_id="a" * 32):
    artifact = tmp_path / "sam3-run.pt"
    sam3_sidecar = tmp_path / "sam3-run.pt.sam3_meta.json"
    artifact.write_bytes(b"x")
    sam3_sidecar.write_text(json.dumps({"publish_attempt_id": attempt_id}))
    return artifact, sam3_sidecar


def test_cleanup_keeps_owned_tiling_sidecar_with_profiles(tmp_path):
    """Review m1: user calibration beside the artifact outlives a failed publish."""
    artifact, sam3_sidecar = _owned_pair(tmp_path)
    profile = {
        "id": "p-1",
        "name": "Calibrated",
        "note": "",
        "settings": {},
        "measurement": {},
    }
    tiling = sidecar_path(artifact)
    write_slice_meta(
        artifact,
        {
            "schema_version": 3,
            "model_family": "sam3",
            "training_geometry": {},
            "primary_profile_id": "p-1",
            "profiles": [profile],
        },
    )
    staged = tiling.with_name(tiling.name + ".tmp")
    staged.write_text("{}")
    publish._cleanup_attempt(
        artifact_path=artifact,
        sidecar_path=sam3_sidecar,
        control_dir=None,
        attempt_id="a" * 32,
    )
    assert not artifact.exists() and not sam3_sidecar.exists()
    assert json.loads(tiling.read_text())["profiles"] == [profile]
    assert not staged.exists()


def test_cleanup_keeps_unreadable_tiling_sidecar(tmp_path):
    """Cannot prove it is profile-free, so it is never deleted."""
    artifact, sam3_sidecar = _owned_pair(tmp_path)
    tiling = sidecar_path(artifact)
    tiling.write_text("{not json")
    publish._cleanup_attempt(
        artifact_path=artifact,
        sidecar_path=sam3_sidecar,
        control_dir=None,
        attempt_id="a" * 32,
    )
    assert tiling.exists()


def test_write_failure_after_staging_leaves_no_tmp(tmp_path, monkeypatch):
    """Review m2: a failed replace() must not strand <sidecar>.tmp beside the artifact."""
    artifact = tmp_path / "sam3-run.pt"
    artifact.write_bytes(b"x")

    def half_write(model_path, meta):
        target = sidecar_path(model_path)
        target.with_name(target.name + ".tmp").write_text(json.dumps(meta))
        raise OSError("replace failed")

    monkeypatch.setattr(publish_worker, "write_slice_meta", half_write)
    assert publish_worker._write_tiling_sidecar(artifact, FULL_MANIFEST) is None
    tiling = sidecar_path(artifact)
    assert not tiling.exists()
    assert not tiling.with_name(tiling.name + ".tmp").exists()


# --- Review m8 / code review #1: the real publish_sam3_artifact wiring ------


def _publish_e2e(tmp_path, manifest, **kwargs):
    import torch  # noqa: F401  (the atomic fixture saves real tensors)

    from tests.test_sam3_publish_atomic import _inputs

    _base, _adapters, base_path, adapters_path = _inputs(tmp_path)
    return publish_worker.publish_sam3_artifact(
        run_id="run-1",
        adapters_path=adapters_path,
        base_checkpoint=base_path,
        # What the child really receives: the parent payload, JSON round-tripped.
        build_manifest=json.loads(json.dumps(_child_manifest(manifest))),
        params=Sam3LoraParams(
            prompt="ant", rank=2, alpha=4, label_quality_acknowledged=True
        ),
        source_fingerprint="fp1",
        models_root=tmp_path / "models",
        **kwargs,
    )


def test_publish_e2e_writes_v3_sam3_sidecar(tmp_path):
    artifact, sam3_sidecar = _publish_e2e(tmp_path, FULL_MANIFEST)
    assert artifact.exists() and sam3_sidecar.exists()
    doc = json.loads(sidecar_path(artifact).read_text())
    assert doc["schema_version"] == 3 and doc["model_family"] == "sam3"
    geometry = doc["training_geometry"]
    assert geometry["imgsz"] == publish_worker.PREDICTOR_IMGSZ == 1008
    assert geometry["tile_px_set"] == [[1940, 1940], [970, 970]]
    assert geometry["overlap"] == 0.25
    assert geometry["min_area_ratio"] == 0.3
    assert geometry["geometry_mode"] == "auto_object"
    assert geometry["prefill_object_tile_fraction"] == 0.04125
    assert "object_tile_fraction" not in geometry  # multi-scale: no bare scalar
    meta = read_tiling_meta(artifact)
    assert meta.source == "slice_meta" and meta.operating_fraction == 0.04125


def test_publish_e2e_survives_tiling_write_failure(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(publish_worker, "write_slice_meta", boom)
    artifact, sam3_sidecar = _publish_e2e(tmp_path, FULL_MANIFEST)
    assert artifact.exists() and sam3_sidecar.exists()
    tiling = sidecar_path(artifact)
    assert not tiling.exists()
    assert not tiling.with_name(tiling.name + ".tmp").exists()
    # Readers fall back to the legacy sidecar.
    assert read_tiling_meta(artifact).source == "sam3_meta"


def test_publish_e2e_promotion_failure_leaves_no_tiling_sidecar(tmp_path, monkeypatch):
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
        _publish_e2e(tmp_path, FULL_MANIFEST)
    out_dir = tmp_path / "models" / "sam3_finetuned"
    assert not (out_dir / "run-1.pt.slice_meta.json").exists()
    assert not (out_dir / "run-1.pt.slice_meta.json.tmp").exists()
