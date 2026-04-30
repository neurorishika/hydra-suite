from __future__ import annotations

import numpy as np

from hydra_suite.training.contracts import (
    AugmentationProfile,
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)
from hydra_suite.training.runner import (
    _apply_tiny_augmentation,
    build_ultralytics_command,
)


def test_augmentation_args_passed_to_command():
    """Augmentation args dict entries appear as CLI flags."""
    spec = TrainingRunSpec(
        role=TrainingRole.OBB_DIRECT,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="yolo26s-obb.pt",
        hyperparams=TrainingHyperParams(epochs=10, imgsz=640),
        augmentation_profile=AugmentationProfile(
            enabled=True,
            args={"flipud": 0.3, "mosaic": 0.0, "mixup": 0.1},
        ),
    )
    cmd = build_ultralytics_command(spec, "/tmp/run")
    cmd_str = " ".join(cmd)
    assert "flipud=0.3" in cmd_str
    assert "mosaic=0.0" in cmd_str
    assert "mixup=0.1" in cmd_str


def test_augmentation_disabled_skips_args():
    """When enabled=False, no augmentation args are emitted."""
    spec = TrainingRunSpec(
        role=TrainingRole.OBB_DIRECT,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="yolo26s-obb.pt",
        hyperparams=TrainingHyperParams(epochs=10, imgsz=640),
        augmentation_profile=AugmentationProfile(
            enabled=False,
            args={"flipud": 0.3},
        ),
    )
    cmd = build_ultralytics_command(spec, "/tmp/run")
    cmd_str = " ".join(cmd)
    assert "flipud" not in cmd_str


def test_tiny_augmentation_monochrome_enforces_equal_channels():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[..., 0] = 220
    img[..., 1] = 80
    img[..., 2] = 20

    out = _apply_tiny_augmentation(
        img,
        augment=True,
        profile=AugmentationProfile(enabled=True, monochrome=True),
    )

    assert out.shape == img.shape
    assert np.array_equal(out[..., 0], out[..., 1])
    assert np.array_equal(out[..., 1], out[..., 2])
    assert out.shape == img.shape
    assert np.array_equal(out[..., 0], out[..., 1])
    assert np.array_equal(out[..., 1], out[..., 2])
