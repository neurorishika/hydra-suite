"""SEMANTIC_SAM3 must fork to publish_sam3_model, never publish_trained_model."""

import hydra_suite.training.service as svc
from hydra_suite.training.contracts import (
    Sam3LoraParams,
    SourceDataset,
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)


def _spec(role=TrainingRole.SEMANTIC_SAM3):
    return TrainingRunSpec(
        role=role,
        source_datasets=[SourceDataset(path="/tmp/x", level="polygon")],
        derived_dataset_dir="/tmp/d",
        base_model="sam3",
        hyperparams=TrainingHyperParams(),
        sam3_params=Sam3LoraParams(prompt="ant"),
    )


def test_sam3_role_forks_to_publish_sam3_model(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(
        svc, "publish_trained_model", lambda *a, **k: called.setdefault("yolo", True)
    )
    monkeypatch.setattr(
        svc,
        "publish_sam3_model",
        lambda **k: (called.setdefault("sam3", True), ("key", "/tmp/a.pt"))[1],
    )
    monkeypatch.setattr(svc, "ensure_checkpoint", lambda *a, **k: "/tmp/base.pt")
    key, path = svc._publish_training_artifacts(
        spec=_spec(),
        artifact_paths=[str(tmp_path / "adapters.pt")],
        publish_metadata={},
        run_id="r1",
        dataset_fingerprint_value="fp1",
    )
    assert "sam3" in called
    # publish_trained_model raises "Unsupported publish role" for this role.
    assert "yolo" not in called
    assert path == "/tmp/a.pt"


def test_other_roles_still_use_the_yolo_publisher(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(
        svc,
        "publish_trained_model",
        lambda *a, **k: (called.setdefault("yolo", True), ("k", "/tmp/y.pt"))[1],
    )
    monkeypatch.setattr(
        svc, "publish_sam3_model", lambda **k: called.setdefault("sam3", True)
    )
    svc._publish_training_artifacts(
        spec=_spec(TrainingRole.SEGMENT_DIRECT),
        artifact_paths=[str(tmp_path / "best.pt")],
        publish_metadata={},
        run_id="r1",
        dataset_fingerprint_value="fp1",
    )
    assert "yolo" in called and "sam3" not in called


def test_builder_geometry_reaches_the_sidecar(monkeypatch, tmp_path):
    import json

    derived = tmp_path / "derived"
    derived.mkdir()
    (derived / "build_manifest.json").write_text(
        json.dumps({"tile_px": 1007, "reference_body_px": 55.4})
    )
    seen = {}
    monkeypatch.setattr(
        svc, "publish_sam3_model", lambda **k: (seen.update(k), ("key", "/tmp/a.pt"))[1]
    )
    monkeypatch.setattr(svc, "ensure_checkpoint", lambda *a, **k: "/tmp/base.pt")
    spec = _spec()
    spec.derived_dataset_dir = str(derived)
    svc._publish_training_artifacts(
        spec=spec,
        artifact_paths=[str(tmp_path / "adapters.pt")],
        publish_metadata={"species": "ant"},
        run_id="r1",
        dataset_fingerprint_value="fp1",
    )
    assert seen["build_manifest"]["tile_px"] == 1007
    assert seen["source_fingerprint"] == "fp1"
