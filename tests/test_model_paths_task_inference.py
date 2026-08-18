"""Unit tests for checkpoint-task inference + registry task backfill helpers."""

import json
import os

import pytest

from hydra_suite.core.inference.model_paths import (
    get_yolo_model_registered_task,
    infer_checkpoint_imgsz,
    infer_checkpoint_task,
    register_yolo_model_task,
)


@pytest.fixture
def model_repo(tmp_path, monkeypatch):
    data_dir = tmp_path / "hydra-data"
    monkeypatch.setenv("HYDRA_DATA_DIR", str(data_dir))
    models_root = data_dir / "models"
    obb_dir = models_root / "obb"
    obb_dir.mkdir(parents=True, exist_ok=True)
    return models_root, obb_dir


def _write_registry(models_root, entries):
    (models_root / "model_registry.json").write_text(
        json.dumps({"schema_version": 2, "entries": entries}), encoding="utf-8"
    )


def test_infer_checkpoint_task_skips_stub_files_fast(model_repo):
    """Sub-64KB files (test stubs, garbage) are rejected without importing
    ultralytics — the heavy path must not run for them."""
    _models_root, obb_dir = model_repo
    stub = obb_dir / "stub.pt"
    stub.write_text("stub model", encoding="utf-8")
    assert os.path.getsize(stub) < 64 * 1024
    assert infer_checkpoint_task("obb/stub.pt") == ""


def test_infer_checkpoint_task_missing_path(model_repo):
    assert infer_checkpoint_task("") == ""
    assert infer_checkpoint_task("obb/does_not_exist.pt") == ""


def test_register_and_read_registered_task_roundtrip(model_repo):
    models_root, obb_dir = model_repo
    (obb_dir / "m.pt").write_text("x" * (65 * 1024), encoding="utf-8")
    _write_registry(
        models_root,
        {"obb/m.pt": {"task_family": "obb", "usage_role": "obb_direct"}},
    )
    assert get_yolo_model_registered_task("obb/m.pt") == ""
    assert register_yolo_model_task("obb/m.pt", "detect") is True
    assert get_yolo_model_registered_task("obb/m.pt") == "detect"
    # The task survives a registry reload (normalization keeps unknown keys).
    assert get_yolo_model_registered_task("obb/m.pt") == "detect"


def test_register_task_never_overwrites_explicit_value(model_repo):
    models_root, obb_dir = model_repo
    (obb_dir / "m.pt").write_text("x" * (65 * 1024), encoding="utf-8")
    _write_registry(
        models_root,
        {"obb/m.pt": {"task": "segment", "usage_role": "obb_direct"}},
    )
    assert register_yolo_model_task("obb/m.pt", "detect") is False
    assert get_yolo_model_registered_task("obb/m.pt") == "segment"


def test_register_task_requires_known_task(model_repo):
    models_root, obb_dir = model_repo
    (obb_dir / "m.pt").write_text("x" * (65 * 1024), encoding="utf-8")
    _write_registry(models_root, {"obb/m.pt": {"usage_role": "obb_direct"}})
    assert register_yolo_model_task("obb/m.pt", "not-a-task") is False
    assert register_yolo_model_task("obb/m.pt", "") is False
    assert get_yolo_model_registered_task("obb/m.pt") == ""


def test_register_task_requires_registered_model(model_repo):
    _models_root, _obb_dir = model_repo
    assert register_yolo_model_task("obb/unknown.pt", "obb") is False


def test_infer_checkpoint_imgsz_skips_stub_files(model_repo):
    _models_root, obb_dir = model_repo
    stub = obb_dir / "stub.pt"
    stub.write_text("stub model", encoding="utf-8")
    assert infer_checkpoint_imgsz("obb/stub.pt") == 0
    assert infer_checkpoint_imgsz("") == 0
    assert infer_checkpoint_imgsz("obb/does_not_exist.pt") == 0
