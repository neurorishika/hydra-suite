"""Unit tests for _slice_geometry_for_publish wiring (spec Acceptance #5).

Verifies that the training service reads slice_geometry out of the derived
dataset's manifest.json for OBB_DIRECT runs, and returns None for every other
case (non-OBB_DIRECT role, missing key, missing manifest) so publish behavior
stays byte-identical when slicing is disabled or irrelevant.
"""

import json

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


def test_returns_geometry_for_obb_direct_with_manifest(tmp_path):
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
    spec = _make_spec(role=TrainingRole.OBB_DIRECT, derived_dataset_dir=str(tmp_path))
    result = _slice_geometry_for_publish(spec)
    assert result is not None
    assert result["reference_body_px"] == 42.0


def test_returns_none_for_non_obb_direct_role(tmp_path):
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
