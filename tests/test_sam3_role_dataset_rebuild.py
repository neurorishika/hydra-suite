"""SAM3 role datasets are rebuilt from each run's merged source corpus."""

from __future__ import annotations

import hydra_suite.training.service as svc
from hydra_suite.training.contracts import (
    DatasetBuildResult,
    TrainingRole,
    ValidationReport,
)
from hydra_suite.training.service import TrainingOrchestrator


def _fake_prepare(role, merged_obb_dataset_dir, role_output_root, *a, **kw):
    return DatasetBuildResult(dataset_dir=str(role_output_root))


def _patch_builders(monkeypatch):
    monkeypatch.setattr(svc, "prepare_role_dataset", _fake_prepare)
    monkeypatch.setattr(
        svc, "validate_role_dataset", lambda *a, **k: ValidationReport(valid=True)
    )


def test_sam3_rebuild_accepts_a_new_merged_dataset(tmp_path, monkeypatch):
    """Every preparation builds a fresh merged source directory, so a later
    SAM3 run must not reject the new corpus just because its path changed."""

    _patch_builders(monkeypatch)
    orch = TrainingOrchestrator(tmp_path)

    merged_a = tmp_path / "combined_polygon_a"
    merged_b = tmp_path / "combined_polygon_b"
    merged_a.mkdir()
    merged_b.mkdir()

    orch.build_role_dataset(TrainingRole.SEMANTIC_SAM3, str(merged_a))
    result = orch.build_role_dataset(TrainingRole.SEMANTIC_SAM3, str(merged_b))

    assert result.dataset_dir


def test_other_roles_are_unaffected_by_sam3_rebuilds(tmp_path, monkeypatch):
    _patch_builders(monkeypatch)
    orch = TrainingOrchestrator(tmp_path)

    merged_a = tmp_path / "merged_a"
    merged_b = tmp_path / "merged_b"
    merged_a.mkdir()
    merged_b.mkdir()

    orch.build_role_dataset(TrainingRole.SEMANTIC_SAM3, str(merged_a))
    orch.build_role_dataset(TrainingRole.SEQ_CROP_OBB, str(merged_b))
