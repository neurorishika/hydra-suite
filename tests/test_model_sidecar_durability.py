import json
from pathlib import Path

import pytest

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


def test_creates_destination_directory_when_missing(tmp_path):
    """Kills a mutation dropping dst_sidecar.parent.mkdir(...)."""
    src = tmp_path / "a.pt"
    src.write_text("w", encoding="utf-8")
    Path(str(src) + ".slice_meta.json").write_text("{}", encoding="utf-8")

    dst_dir = tmp_path / "not_yet_created"
    dst = dst_dir / "b.pt"
    assert not dst_dir.exists()

    copy_model_metadata_sidecars(src, dst)

    assert dst_dir.exists()
    assert Path(str(dst) + ".slice_meta.json").exists()


def test_v2meta_uses_suffix_replace_not_append_convention(tmp_path):
    """Kills a mutation that copies .v2meta.json with the wrong (append) name,
    or drops the .v2meta.json pair entirely."""
    src = tmp_path / "a.pt"
    src.write_text("w", encoding="utf-8")
    # Correct convention: a.pt -> a.v2meta.json (suffix REPLACE).
    src.with_suffix(".v2meta.json").write_text('{"real": true}', encoding="utf-8")
    # A decoy in the wrong (append) convention must never be produced.
    wrong_convention = Path(str(src) + ".v2meta.json")
    assert not wrong_convention.exists()

    dst = tmp_path / "out" / "b.pt"
    dst.parent.mkdir()
    dst.write_text("w", encoding="utf-8")
    copy_model_metadata_sidecars(src, dst)

    correct = dst.with_suffix(".v2meta.json")
    assert correct.exists()
    assert json.loads(correct.read_text(encoding="utf-8")) == {"real": True}
    assert not Path(str(dst) + ".v2meta.json").exists()


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


def test_trackerkit_import_carries_sidecar_and_stamps_registry(tmp_path, monkeypatch):
    """F4: TrackerKit's 'Add model' import copies the calibration sidecar and
    records slice_profiles in the registry entry, matching the copied sidecar."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QDialog

    _app = QApplication.instance() or QApplication([])

    from tests.test_main_window_config_persistence import _make_main_window

    data_dir = tmp_path / "hydra-data"
    monkeypatch.setenv("HYDRA_DATA_DIR", str(data_dir))
    models_root = data_dir / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    (models_root / "model_registry.json").write_text(
        json.dumps({"schema_version": 2, "entries": {}}, indent=2),
        encoding="utf-8",
    )

    # Bypass the modal dialog: accept it with whatever defaults were populated
    # from the source filename, so the import proceeds without user input.
    monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.Accepted)

    window = _make_main_window(monkeypatch)

    src = tmp_path / "ant_calibrated.pt"
    src.write_bytes(b"weights")
    Path(str(src) + ".slice_meta.json").write_text(
        json.dumps(META, indent=2), encoding="utf-8"
    )

    rel_path = window._import_yolo_model_to_repository(str(src))
    assert rel_path is not None

    from hydra_suite.core.inference.model_paths import (
        get_yolo_model_metadata,
        get_yolo_model_repository_directory,
    )

    entry = get_yolo_model_metadata(rel_path)
    assert entry is not None
    assert "slice_profiles" in entry
    sidecar_count = entry["slice_profiles"]["count"]

    stored_filename = entry.get("stored_filename")
    assert stored_filename
    stored_path = Path(get_yolo_model_repository_directory()) / stored_filename
    dst_meta = read_slice_meta(stored_path)
    assert dst_meta is not None
    assert len(dst_meta["profiles"]) == sidecar_count == 1
    assert entry["slice_profiles"]["primary_profile_id"] == "p"
