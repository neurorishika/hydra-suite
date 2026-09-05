import json
from pathlib import Path

from hydra_suite.core.inference.model_paths import copy_model_metadata_sidecars
from hydra_suite.core.inference.slice_meta import read_slice_meta
from hydra_suite.training.contracts import TrainingRole

META = {
    "schema_version": 2,
    "training_geometry": {"geometry_mode": "auto_model"},
    "primary_profile_id": "p",
    "profiles": [{"id": "p", "name": "P", "settings": {"enabled": True}}],
}


def test_copies_every_sidecar_suffix(tmp_path):
    src = tmp_path / "a.pt"
    src.write_text("w", encoding="utf-8")
    for suffix in (".slice_meta.json", ".canonical_meta.json", ".runtime_meta.json"):
        Path(str(src) + suffix).write_text("{}", encoding="utf-8")
    src.with_suffix(".v2meta.json").write_text("{}", encoding="utf-8")

    dst = tmp_path / "out" / "b.pt"
    dst.parent.mkdir()
    dst.write_text("w", encoding="utf-8")
    copy_model_metadata_sidecars(src, dst)

    for suffix in (".slice_meta.json", ".canonical_meta.json", ".runtime_meta.json"):
        assert Path(str(dst) + suffix).exists(), suffix
    assert dst.with_suffix(".v2meta.json").exists()


def test_absent_sidecars_are_not_an_error(tmp_path):
    src = tmp_path / "a.pt"
    src.write_text("w", encoding="utf-8")
    dst = tmp_path / "b.pt"
    dst.write_text("w", encoding="utf-8")
    copy_model_metadata_sidecars(src, dst)  # must not raise


def test_publish_without_slice_geometry_keeps_profiles(tmp_path, monkeypatch):
    """A calibrated-then-registered model must not lose its profiles."""
    data_dir = tmp_path / "hydra-data"
    monkeypatch.setenv("HYDRA_DATA_DIR", str(data_dir))

    from hydra_suite.training import model_publish as mp

    src = tmp_path / "src_weights.pt"
    src.write_bytes(b"weights")
    Path(str(src) + ".slice_meta.json").write_text(
        json.dumps(META, indent=2), encoding="utf-8"
    )

    key, stored = mp.publish_trained_model(
        role=TrainingRole.OBB_DIRECT,
        artifact_path=str(src),
        size="s",
        species="ant",
        model_info="calibrated",
        trained_from_run_id="r1",
        dataset_fingerprint="fp",
        base_model="yolo26s-obb.pt",
        slice_geometry=None,
    )

    stored_path = Path(stored)
    dst_meta = read_slice_meta(stored_path)
    assert dst_meta is not None
    profile_ids = [p["id"] for p in dst_meta["profiles"]]
    assert "p" in profile_ids

    reg = mp.load_model_registry()
    entry = reg["entries"][key]
    assert entry["slice_profiles"] == {
        "count": 1,
        "primary_profile_id": "p",
        "names": ["P"],
    }


def test_no_slice_geometry_and_no_sidecar_writes_no_sidecar(tmp_path, monkeypatch):
    """Negative case: publishing an uncalibrated source stays sidecar-free."""
    data_dir = tmp_path / "hydra-data"
    monkeypatch.setenv("HYDRA_DATA_DIR", str(data_dir))

    from hydra_suite.training import model_publish as mp

    src = tmp_path / "plain_weights.pt"
    src.write_bytes(b"weights")

    key, stored = mp.publish_trained_model(
        role=TrainingRole.OBB_DIRECT,
        artifact_path=str(src),
        size="s",
        species="ant",
        model_info="plain",
        trained_from_run_id="r2",
        dataset_fingerprint="fp",
        base_model="yolo26s-obb.pt",
        slice_geometry=None,
    )

    stored_path = Path(stored)
    assert not Path(str(stored_path) + ".slice_meta.json").exists()
    reg = mp.load_model_registry()
    entry = reg["entries"][key]
    assert "slice_geometry" not in entry
    assert "slice_meta_sidecar" not in entry
    assert "slice_profiles" not in entry
