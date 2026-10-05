"""Plan (CLI/JSON) parsing and validation for ``sam3.augmentation``."""

from __future__ import annotations

import json

import pytest

from hydra_suite.detectkit.config.training import (
    DetectTrainingPlan,
    TrainingPlanError,
    load_training_plan,
)
from hydra_suite.training.contracts import AugmentationProfile


def _payload(sam3_extra: dict | None = None) -> dict:
    sam3 = {"label_quality_acknowledged": True, "prompt": "ant"}
    if sam3_extra is not None:
        sam3.update(sam3_extra)
    return {
        "version": 1,
        "workspace": "./workspace",
        "sources": [{"path": "./source", "name": "day-1", "level": "polygon"}],
        "class_names": ["ant"],
        "dataset": {"split": {"train": 0.8, "val": 0.2, "test": 0.0}},
        "training": {"device": "0", "seed": 7, "epochs": 12, "batch": 4},
        "roles": [{"role": "semantic_sam3", "imgsz": 1008}],
        "sam3": sam3,
    }


def _load(tmp_path, payload):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_training_plan(path)


def test_sam3_augmentation_loads_and_round_trips(tmp_path):
    aug = {"enabled": True, "fliplr": 0.5, "rot90": 0.5, "brightness": 0.2}
    plan = _load(tmp_path, _payload({"augmentation": aug}))
    profile = plan.sam3_params.augmentation
    assert isinstance(profile, AugmentationProfile)
    assert profile.enabled is True
    assert (profile.fliplr, profile.rot90, profile.brightness) == (0.5, 0.5, 0.2)

    again = DetectTrainingPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    assert again.sam3_params.augmentation == profile


def test_sam3_augmentation_default_disabled(tmp_path):
    plan = _load(tmp_path, _payload())
    assert plan.sam3_params.augmentation.enabled is False


@pytest.mark.parametrize(
    "aug",
    [
        {"enabled": True, "fliplr": 2},
        {"enabled": True, "args": {"mosaic": 1}},
        {"enabled": True, "canonical_aug": True},
    ],
)
def test_sam3_augmentation_validate_rejects(tmp_path, aug):
    key = next(k for k in ("fliplr", "args", "canonical_aug") if k in aug)
    # load_training_plan runs validate(), so the error surfaces at load.
    with pytest.raises(TrainingPlanError, match=rf"sam3\.augmentation\.{key}"):
        _load(tmp_path, _payload({"augmentation": aug}))


def test_sam3_augmentation_bad_type_rejected_at_load(tmp_path):
    with pytest.raises(TrainingPlanError):
        _load(tmp_path, _payload({"augmentation": {"fliplr": "yes"}}))


def test_sam3_augmentation_unknown_key_rejected_at_load(tmp_path):
    with pytest.raises(TrainingPlanError):
        _load(tmp_path, _payload({"augmentation": {"mosaic": 1}}))


def test_training_augmentation_rot90_loads(tmp_path):
    payload = _payload()
    payload["training"]["augmentation"] = {"enabled": True, "rot90": 0.5}
    plan = _load(tmp_path, payload)
    assert plan.augmentation_profile.rot90 == 0.5


def test_sam3_augmentation_null_is_disabled(tmp_path):
    plan = _load(tmp_path, _payload({"augmentation": None}))
    assert plan.sam3_params.augmentation.enabled is False


def test_sam3_augmentation_empty_mapping_keeps_profile_defaults(tmp_path):
    plan = _load(tmp_path, _payload({"augmentation": {}}))
    assert plan.sam3_params.augmentation == AugmentationProfile()
