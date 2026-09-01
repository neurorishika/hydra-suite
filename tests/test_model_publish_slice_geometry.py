import json
from pathlib import Path

import pytest

import hydra_suite.training.model_publish as mp
from hydra_suite.core.inference.slice_meta import (
    available_slice_profiles,
    primary_slice_profile,
    read_slice_meta,
    slice_meta_to_panel_values,
    write_slice_meta,
)
from hydra_suite.training.contracts import TrainingRole


@pytest.mark.parametrize(
    "role,base_model",
    [
        (TrainingRole.OBB_DIRECT, "yolo26s-obb.pt"),
        (TrainingRole.DETECT_DIRECT, "yolo26s.pt"),
        (TrainingRole.SEGMENT_DIRECT, "yolo26s-seg.pt"),
    ],
)
def test_all_direct_detector_roles_publish_slice_geometry(
    tmp_path, monkeypatch, role, base_model
):
    monkeypatch.setattr(mp, "get_models_root", lambda: tmp_path)
    src = tmp_path / f"{role.value}.pt"
    src.write_bytes(b"fake-weights")
    geometry = {"geometry_mode": "auto_object", "reference_body_px": 42.0}

    key, stored = mp.publish_trained_model(
        role=role,
        artifact_path=str(src),
        size="s",
        species="ant",
        model_info="sliced",
        trained_from_run_id="r1",
        dataset_fingerprint="fp",
        base_model=base_model,
        slice_geometry=geometry,
    )

    sidecar = Path(stored).with_suffix(Path(stored).suffix + ".slice_meta.json")
    assert json.loads(sidecar.read_text()) == geometry
    assert mp.load_model_registry()["entries"][key]["slice_geometry"] == geometry


def test_slice_geometry_written_as_sidecar_and_registry(tmp_path, monkeypatch):
    # Redirect the models root into tmp_path using the established pattern from
    # tests/test_classkit_publish.py: monkeypatch get_models_root directly rather
    # than relying on the _project_root/__module__ override trick, which is
    # more fragile (depends on internal `_use_project_root_override` machinery).
    monkeypatch.setattr(mp, "get_models_root", lambda: tmp_path)

    src = tmp_path / "weights.pt"
    src.write_bytes(b"fake-weights")
    geom = {
        "geometry_mode": "auto_object",
        "target_sizes": [200.0, 300.0],
        "reference_body_px": 42.0,
    }

    key, stored = mp.publish_trained_model(
        role=TrainingRole.OBB_DIRECT,
        artifact_path=str(src),
        size="s",
        species="ant",
        model_info="sliced",
        trained_from_run_id="r1",
        dataset_fingerprint="fp",
        base_model="yolo26s-obb.pt",
        slice_geometry=geom,
    )
    stored_path = Path(stored)
    sidecar = stored_path.with_suffix(stored_path.suffix + ".slice_meta.json")
    assert sidecar.exists()
    assert json.loads(sidecar.read_text())["reference_body_px"] == 42.0
    reg = mp.load_model_registry()
    assert reg["entries"][key]["slice_geometry"]["geometry_mode"] == "auto_object"
    assert reg["entries"][key]["slice_meta_sidecar"] == sidecar.name


def test_published_sidecar_is_read_back_by_read_slice_meta(tmp_path, monkeypatch):
    """Round trip: the name publish writes MUST be the name TrackerKit reads.

    Guards the publish->TrackerKit handoff end-to-end. The per-side unit tests
    each pinned their own filename convention and disagreed (writer replaced the
    .pt suffix, reader appended to it), so the sidecar was inert in production
    while both suites stayed green. read_slice_meta is the actual reader path.
    """
    monkeypatch.setattr(mp, "get_models_root", lambda: tmp_path)
    src = tmp_path / "weights.pt"
    src.write_bytes(b"fake-weights")
    geom = {
        "geometry_mode": "auto_object",
        "target_sizes": [200.0, 300.0],
        "reference_body_px": 42.0,
    }
    _key, stored = mp.publish_trained_model(
        role=TrainingRole.OBB_DIRECT,
        artifact_path=str(src),
        size="s",
        species="ant",
        model_info="sliced",
        trained_from_run_id="r1",
        dataset_fingerprint="fp",
        base_model="yolo26s-obb.pt",
        slice_geometry=geom,
    )
    meta = read_slice_meta(stored)
    assert meta is not None
    assert meta["reference_body_px"] == 42.0
    assert meta["geometry_mode"] == "auto_object"


def test_no_slice_geometry_writes_no_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "get_models_root", lambda: tmp_path)
    src = tmp_path / "w2.pt"
    src.write_bytes(b"x")
    key, stored = mp.publish_trained_model(
        role=TrainingRole.OBB_DIRECT,
        artifact_path=str(src),
        size="s",
        species="ant",
        model_info="plain",
        trained_from_run_id="r2",
        dataset_fingerprint="fp",
        base_model="yolo26s-obb.pt",
    )
    stored_path = Path(stored)
    assert not stored_path.with_suffix(stored_path.suffix + ".slice_meta.json").exists()
    reg = mp.load_model_registry()
    assert "slice_geometry" not in reg["entries"][key]
    assert "slice_meta_sidecar" not in reg["entries"][key]


def test_calibrated_profiles_apply_primary_without_losing_training_geometry(tmp_path):
    model = tmp_path / "weights.pt"
    model.write_bytes(b"weights")
    meta = {
        "schema_version": 2,
        "training_geometry": {"geometry_mode": "auto_model", "imgsz": 640},
        "primary_profile_id": "recall",
        "profiles": [
            {
                "id": "recall",
                "name": "High recall",
                "settings": {
                    "geometry_mode": "auto_object",
                    "object_tile_fraction": 0.4,
                    "overlap": 0.3,
                    "trained_body_px": 80,
                    "slice_width": 0,
                    "slice_height": 0,
                    "confidence_threshold": 0.2,
                },
            },
            {
                "id": "fast",
                "name": "Fast scan",
                "settings": {"geometry_mode": "auto_model", "overlap": 0.1},
            },
        ],
    }
    write_slice_meta(model, meta)

    loaded = read_slice_meta(model)
    assert loaded == meta
    assert [profile["id"] for profile in available_slice_profiles(loaded)] == [
        "recall",
        "fast",
    ]
    assert primary_slice_profile(loaded)["name"] == "High recall"
    primary = slice_meta_to_panel_values(loaded)
    fast = slice_meta_to_panel_values(loaded, "fast")
    assert primary["profile_id"] == "recall"
    assert primary["confidence_threshold"] == 0.2
    assert fast["profile_name"] == "Fast scan"
    assert fast["geometry_mode"] == "auto_model"


def test_missing_profile_falls_back_to_primary_and_legacy_sidecar_still_loads():
    meta = {
        "schema_version": 2,
        "training_geometry": {"geometry_mode": "custom", "slice_width": 700},
        "primary_profile_id": "saved",
        "profiles": [
            {"id": "saved", "name": "Saved", "settings": {"overlap": 0.25}},
        ],
    }
    assert slice_meta_to_panel_values(meta, "removed")["profile_id"] == "saved"
    legacy = {
        "geometry_mode": "auto_object",
        "target_sizes": [300.0],
        "imgsz": 640,
        "reference_body_px": 42.0,
    }
    values = slice_meta_to_panel_values(legacy)
    assert values["profile_id"] is None
    assert values["object_tile_fraction"] == 300.0 / 640.0
