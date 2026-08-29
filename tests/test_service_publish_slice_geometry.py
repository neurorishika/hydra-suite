"""Unit tests for _slice_geometry_for_publish wiring (spec Acceptance #5).

Verifies that the training service reads slice_geometry out of the derived
dataset's manifest.json for every direct detector role, and returns None for
other roles or malformed/missing manifests.
"""

import json

import pytest

from hydra_suite.trackerkit.gui.orchestrators.config import ConfigOrchestrator
from hydra_suite.training.contracts import (
    SourceDataset,
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)
from hydra_suite.training.service import _slice_geometry_for_publish


def _make_spec(*, role: TrainingRole, derived_dataset_dir: str) -> TrainingRunSpec:
    return TrainingRunSpec(
        role=role,
        source_datasets=[SourceDataset(path="/tmp/src", source_type="yolo_obb")],
        derived_dataset_dir=derived_dataset_dir,
        base_model="yolo26s-obb.pt",
        hyperparams=TrainingHyperParams(),
    )


@pytest.mark.parametrize(
    "role",
    [
        TrainingRole.OBB_DIRECT,
        TrainingRole.DETECT_DIRECT,
        TrainingRole.SEGMENT_DIRECT,
    ],
)
def test_returns_geometry_for_all_direct_detector_roles(tmp_path, role):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "type": "sliced_obb",
                "slice_geometry": {
                    "geometry_mode": "auto_object",
                    "reference_body_px": 42.0,
                },
            }
        )
    )
    spec = _make_spec(role=role, derived_dataset_dir=str(tmp_path))
    result = _slice_geometry_for_publish(spec)
    assert result is not None
    assert result["reference_body_px"] == 42.0


def test_returns_none_for_non_direct_detector_role(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "type": "sliced_obb",
                "slice_geometry": {
                    "geometry_mode": "auto_object",
                    "reference_body_px": 42.0,
                },
            }
        )
    )
    spec = _make_spec(role=TrainingRole.SEQ_DETECT, derived_dataset_dir=str(tmp_path))
    assert _slice_geometry_for_publish(spec) is None


def test_returns_none_when_manifest_has_no_slice_geometry(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"type": "merged_obb"}))
    spec = _make_spec(role=TrainingRole.OBB_DIRECT, derived_dataset_dir=str(tmp_path))
    assert _slice_geometry_for_publish(spec) is None


def test_returns_none_when_manifest_missing(tmp_path):
    spec = _make_spec(role=TrainingRole.OBB_DIRECT, derived_dataset_dir=str(tmp_path))
    assert _slice_geometry_for_publish(spec) is None


@pytest.mark.parametrize(
    "task_family,usage_role",
    [
        ("obb", "obb_direct"),
        ("detect", "detect_direct"),
        ("segment", "segment_direct"),
    ],
)
def test_direct_model_filter_accepts_each_direct_published_role(
    task_family, usage_role
):
    assert ConfigOrchestrator._yolo_model_matches_filter(
        {"task_family": task_family, "usage_role": usage_role},
        task_family={"obb", "detect", "segment"},
        usage_role={"obb_direct", "detect_direct", "segment_direct"},
    )
    assert not ConfigOrchestrator._yolo_model_matches_filter(
        {"task_family": "detect", "usage_role": "seq_detect"},
        task_family={"obb", "detect", "segment"},
        usage_role={"obb_direct", "detect_direct", "segment_direct"},
    )
