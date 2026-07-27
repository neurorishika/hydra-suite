import json
from pathlib import Path

import hydra_suite.training.model_publish as mp
from hydra_suite.training.contracts import TrainingRole


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
    sidecar = Path(stored).with_suffix(".slice_meta.json")
    assert sidecar.exists()
    assert json.loads(sidecar.read_text())["reference_body_px"] == 42.0
    reg = mp.load_model_registry()
    assert reg["entries"][key]["slice_geometry"]["geometry_mode"] == "auto_object"
    assert reg["entries"][key]["slice_meta_sidecar"] == sidecar.name


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
    assert not Path(stored).with_suffix(".slice_meta.json").exists()
    reg = mp.load_model_registry()
    assert "slice_geometry" not in reg["entries"][key]
    assert "slice_meta_sidecar" not in reg["entries"][key]
