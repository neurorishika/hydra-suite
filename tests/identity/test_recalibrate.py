"""Task 9: recalibrate_artifact — Qt-free refit-in-place helper.

Covers the TINY arch end-to-end (rebuild model, build a val loader over a
plain ImageFolder-style ``val_dir``, refit calibration, rewrite the
checkpoint's calibration_temperature/signature/ece in place, and confirm the
artifact re-parses as "calibrated"). Torchvision-flat and multihead paths are
wired but not exercised end-to-end here (see task-9-report.md).
"""

from pathlib import Path

import cv2
import numpy as np

from hydra_suite.core.individual.classification.backend import (
    ClassifierBackend,
    calibration_status,
)
from hydra_suite.runtime.resolver import ResolvedBackend
from hydra_suite.training.calibration_fit import CalibrationResult, recalibrate_artifact
from hydra_suite.training.torchvision_model import (
    build_torchvision_classifier,
    save_torchvision_checkpoint,
)


def _write_images(cls_dir: Path, n: int, seed: int) -> None:
    cls_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        img = rng.integers(0, 255, size=(48, 96, 3), dtype=np.uint8)
        cv2.imwrite(str(cls_dir / f"{i}.png"), img)


def _make_tiny_checkpoint(tmp_path: Path) -> Path:
    class_names = ["ant", "bee"]
    model = build_torchvision_classifier(
        "tinyclassifier",
        num_classes=len(class_names),
        trainable_layers=-1,
        hidden_layers=1,
        hidden_dim=16,
        input_width=96,
        input_height=48,
    )
    model_path = tmp_path / "model.pth"
    save_torchvision_checkpoint(
        model=model,
        backbone="tinyclassifier",
        class_names=class_names,
        factor_names=["flat"],
        input_size=(48, 96),
        best_val_acc=None,
        history={},
        trainable_layers=-1,
        backbone_lr_scale=1.0,
        monochrome=False,
        path=model_path,
    )
    return model_path


def test_recalibrate_artifact_tiny_end_to_end(tmp_path):
    model_path = _make_tiny_checkpoint(tmp_path)

    val_dir = tmp_path / "val"
    for i, name in enumerate(["ant", "bee"]):
        _write_images(val_dir / name, 6, seed=i)

    result = recalibrate_artifact(str(model_path), str(val_dir))

    assert isinstance(result, CalibrationResult)
    assert len(result.temperatures) == 1
    assert isinstance(result.temperatures[0], float)
    assert result.signature and isinstance(result.signature, str)
    assert len(result.ece_before) == 1
    assert len(result.ece_after) == 1

    backend = ClassifierBackend(str(model_path), ResolvedBackend("torch", "cpu", False))
    try:
        meta = backend.metadata
    finally:
        backend.close()

    assert calibration_status(meta, None) == "calibrated"
    assert meta.calibration_temperature is not None
    assert meta.calibration_signature == result.signature


def test_recalibrate_artifact_does_not_change_weights(tmp_path):
    import torch

    model_path = _make_tiny_checkpoint(tmp_path)
    before = torch.load(str(model_path), map_location="cpu", weights_only=False)
    before_state = {k: v.clone() for k, v in before["model_state_dict"].items()}

    val_dir = tmp_path / "val"
    for i, name in enumerate(["ant", "bee"]):
        _write_images(val_dir / name, 6, seed=i)

    recalibrate_artifact(str(model_path), str(val_dir))

    after = torch.load(str(model_path), map_location="cpu", weights_only=False)
    for k, v in before_state.items():
        assert torch.equal(v, after["model_state_dict"][k])
