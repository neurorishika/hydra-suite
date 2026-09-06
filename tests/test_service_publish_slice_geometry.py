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


# --- R7: the sidecar must carry what the run DID, not what it requested. ---


def test_realised_stamp_overrides_the_requested_balance_block(tmp_path):
    from hydra_suite.training.service import apply_realised_balance_stamp

    run_dir = tmp_path / "runs" / "exp1"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "hydra_scale_balance.json").write_text(
        json.dumps(
            {
                "requested": {"enabled": True, "power": 0.5},
                "applied": {
                    "scale_grouped_batching": False,
                    "multiscale_loss_weighting": False,
                    "power": 0.0,
                },
                "world_size": 4,
                "ddp": True,
                "ddp_opt_out": True,
            }
        )
    )
    geometry = {
        "reference_body_px": 42.0,
        "multiscale_loss_balance": {"enabled": True, "power": 0.5},
    }

    result = apply_realised_balance_stamp(
        geometry, [str(run_dir / "weights" / "best.pt")]
    )

    balance = result["multiscale_loss_balance"]
    assert balance["enabled"] is False
    assert balance["power"] == 0.0
    assert balance["scale_grouped_batching"] is False
    assert balance["world_size"] == 4
    assert balance["requested"] == {"enabled": True, "power": 0.5}
    assert geometry["multiscale_loss_balance"] == {"enabled": True, "power": 0.5}


def test_geometry_is_unchanged_when_no_run_stamp_exists(tmp_path):
    from hydra_suite.training.service import apply_realised_balance_stamp

    geometry = {"multiscale_loss_balance": {"enabled": True, "power": 0.5}}
    weights = tmp_path / "runs" / "exp1" / "weights"
    weights.mkdir(parents=True)

    assert (
        apply_realised_balance_stamp(geometry, [str(weights / "best.pt")]) == geometry
    )


def test_realised_stamp_survives_the_slice_meta_sidecar_round_trip(tmp_path):
    from hydra_suite.core.inference.slice_meta import (
        merge_training_geometry,
        training_geometry,
    )

    geometry = {
        "multiscale_loss_balance": {
            "enabled": False,
            "power": 0.0,
            "scale_grouped_batching": False,
            "world_size": 4,
            "requested": {"enabled": True, "power": 0.5},
        }
    }

    merged = merge_training_geometry(None, geometry)

    assert (
        training_geometry(merged)["multiscale_loss_balance"]
        == geometry["multiscale_loss_balance"]
    )
