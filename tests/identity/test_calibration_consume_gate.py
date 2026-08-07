"""Task 6: CNNConfig.calibration_temperature falls back to artifact metadata.

Consumes Task 5 (`ClassifierMetadata.calibration_temperature`, a per-factor
tuple) at `build_inference_config_from_params`. Flat/scalar consume uses the
first factor's stored temperature when params omit an explicit override;
params always win when they do specify one.
"""

import torch

from hydra_suite.core.inference.config import (
    _resolve_cnn_temperature,
    build_inference_config_from_params,
)
from hydra_suite.training.torchvision_model import save_torchvision_checkpoint


def _save_checkpoint(path: str, *, calibration_temperature=None) -> None:
    model = torch.nn.Linear(4, 3)
    save_torchvision_checkpoint(
        model=model,
        backbone="resnet18",
        class_names=["a", "b", "c"],
        factor_names=["flat"],
        input_size=(64, 64),
        best_val_acc=0.9,
        history=[],
        trainable_layers=0,
        backbone_lr_scale=1.0,
        monochrome=False,
        path=path,
        calibration_temperature=calibration_temperature,
    )


def _minimal_params(cnn_cfg_dict: dict) -> dict:
    return {
        "YOLO_OBB_MODE": "direct",
        "YOLO_OBB_DIRECT_MODEL_PATH": "m.pt",
        "CNN_CLASSIFIERS": [cnn_cfg_dict],
    }


def test_build_config_uses_stored_temperature_when_params_omit_it(tmp_path):
    p = tmp_path / "calibrated.pth"
    _save_checkpoint(str(p), calibration_temperature=[1.7])

    params = _minimal_params({"model_path": str(p), "label": "cnn_identity"})
    cfg = build_inference_config_from_params(params)

    assert len(cfg.cnn_phases) == 1
    assert cfg.cnn_phases[0].calibration_temperature == 1.7


def test_build_config_explicit_temperature_overrides_artifact(tmp_path):
    p = tmp_path / "calibrated.pth"
    _save_checkpoint(str(p), calibration_temperature=[1.7])

    params = _minimal_params(
        {
            "model_path": str(p),
            "label": "cnn_identity",
            "calibration_temperature": 2.5,
        }
    )
    cfg = build_inference_config_from_params(params)

    assert cfg.cnn_phases[0].calibration_temperature == 2.5


def test_build_config_uncalibrated_artifact_defaults_to_one(tmp_path):
    p = tmp_path / "uncalibrated.pth"
    _save_checkpoint(str(p), calibration_temperature=None)

    params = _minimal_params({"model_path": str(p), "label": "cnn_identity"})
    cfg = build_inference_config_from_params(params)

    assert cfg.cnn_phases[0].calibration_temperature == 1.0


def test_resolve_cnn_temperature_nonexistent_model_path_defaults_to_one():
    assert _resolve_cnn_temperature({}, "/no/such/model.pth") == 1.0


def test_resolve_cnn_temperature_explicit_wins_over_bad_path():
    assert (
        _resolve_cnn_temperature({"calibration_temperature": 3.3}, "/no/such/model.pth")
        == 3.3
    )


def test_resolve_cnn_temperature_legacy_temperature_key_still_works(tmp_path):
    p = tmp_path / "calibrated.pth"
    _save_checkpoint(str(p), calibration_temperature=[1.7])
    # Legacy "temperature" key is still an explicit override, taking
    # precedence over the artifact's stored value.
    assert _resolve_cnn_temperature({"temperature": 4.2}, str(p)) == 4.2


def test_resolve_cnn_temperature_reads_first_factor(tmp_path):
    p = tmp_path / "calibrated.pth"
    _save_checkpoint(str(p), calibration_temperature=[1.7])
    assert _resolve_cnn_temperature({}, str(p)) == 1.7
