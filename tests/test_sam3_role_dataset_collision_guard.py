"""The multi-source collision guard must hold at the service layer, not
just in the DetectKit GUI's training dialog.

`build_role_dataset` writes SEMANTIC_SAM3's derived dataset to a single
`out_root` keyed by role only, not by source. Calling it twice with
different sources previously overwrote `train/_annotations.coco.json`
silently, while both sources' images stayed on disk -- only the last
source's annotations survived. The GUI already refuses this up front; this
guard makes the service API itself refuse a programmatic caller doing the
same thing.
"""

from __future__ import annotations

import pytest

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


def test_second_source_for_same_role_fails_loudly(tmp_path, monkeypatch):
    _patch_builders(monkeypatch)
    orch = TrainingOrchestrator(tmp_path)

    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    source_a.mkdir()
    source_b.mkdir()

    orch.build_role_dataset(TrainingRole.SEMANTIC_SAM3, str(source_a))

    with pytest.raises(ValueError, match="one labeled source dataset"):
        orch.build_role_dataset(TrainingRole.SEMANTIC_SAM3, str(source_b))


def test_rebuilding_from_the_same_source_is_allowed(tmp_path, monkeypatch):
    """Re-running the same role from the SAME source (a legitimate retry)
    must not be blocked by the guard."""
    _patch_builders(monkeypatch)
    orch = TrainingOrchestrator(tmp_path)

    source_a = tmp_path / "source_a"
    source_a.mkdir()

    orch.build_role_dataset(TrainingRole.SEMANTIC_SAM3, str(source_a))
    # Should not raise.
    result = orch.build_role_dataset(TrainingRole.SEMANTIC_SAM3, str(source_a))
    assert result.dataset_dir


def test_other_roles_are_entirely_unaffected(tmp_path, monkeypatch):
    """Non-SEMANTIC_SAM3 roles are always built from ONE merged dataset per
    run; repeated calls for a DIFFERENT role, or the same role from a
    changing merged dataset (a legitimate rebuild), must not be guarded."""
    _patch_builders(monkeypatch)
    orch = TrainingOrchestrator(tmp_path)

    merged_a = tmp_path / "merged_a"
    merged_b = tmp_path / "merged_b"
    merged_a.mkdir()
    merged_b.mkdir()

    # SEQ_DETECT rebuilt from a different merged dataset must not raise.
    orch.build_role_dataset(TrainingRole.SEQ_DETECT, str(merged_a))
    orch.build_role_dataset(TrainingRole.SEQ_DETECT, str(merged_b))

    # A different role than SEMANTIC_SAM3, run after SEMANTIC_SAM3 was
    # guarded above, is untouched by SEMANTIC_SAM3's out_root/stamp.
    orch.build_role_dataset(TrainingRole.SEMANTIC_SAM3, str(merged_a))
    orch.build_role_dataset(TrainingRole.SEQ_CROP_OBB, str(merged_b))
