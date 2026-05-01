"""End-to-end smoke test: train a tiny shared-trunk multi-head model on
synthetic data, save a checkpoint, reload it, and check forward shape.

Skipped if torchvision pretrained weights cannot be downloaded.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def _make_synth_dataset(root: Path, n_per_class: int = 3) -> None:
    classes = [
        ("red__blue", (200, 0, 0)),
        ("red__green", (200, 0, 0)),
        ("yellow__blue", (200, 200, 0)),
        ("yellow__green", (200, 200, 0)),
    ]
    for split in ("train", "val"):
        for cls, color in classes:
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                arr = np.zeros((32, 32, 3), dtype=np.uint8)
                arr[:] = color
                Image.fromarray(arr).save(d / f"{cls}_{i}.png")


@pytest.mark.skipif(
    os.environ.get("HYDRA_OFFLINE_TESTS") == "1",
    reason="needs torchvision pretrained weights; skip on offline CI",
)
def test_train_shared_trunk_smoke(tmp_path):
    from hydra_suite.training.contracts import (
        AugmentationProfile,
        CustomCNNParams,
        SourceDataset,
        TrainingHyperParams,
        TrainingRole,
        TrainingRunSpec,
    )
    from hydra_suite.training.runner import run_training
    from hydra_suite.training.torchvision_model import load_torchvision_classifier

    dataset_dir = tmp_path / "dataset"
    _make_synth_dataset(dataset_dir, n_per_class=3)
    run_dir = tmp_path / "run"

    spec = TrainingRunSpec(
        role=TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM_SHARED,
        source_datasets=[SourceDataset(name="synth", path=str(dataset_dir))],
        derived_dataset_dir=str(dataset_dir),
        base_model="resnet18",
        hyperparams=TrainingHyperParams(),
        custom_params=CustomCNNParams(
            backbone="resnet18",
            head_kind="multihead_shared_trunk",
            head_hidden_dim=32,
            head_dropout=0.0,
            epochs=1,
            batch=2,
            lr=1e-3,
            input_size=32,
            label_smoothing=0.1,
            trainable_layers=0,
        ),
        augmentation_profile=AugmentationProfile(),
        device="cpu",
    )
    spec.augmentation_profile.args["class_names_per_factor"] = [
        ["red", "yellow"],
        ["blue", "green"],
    ]
    spec.augmentation_profile.args["factor_names"] = ["tag_1", "tag_2"]

    result = run_training(spec, run_dir)
    assert result.get("artifact_path"), result
    artifact_path = Path(result["artifact_path"])
    assert artifact_path.exists()

    model, ckpt = load_torchvision_classifier(str(artifact_path), device="cpu")
    assert ckpt["head_kind"] == "multihead_shared_trunk"
    assert ckpt["factor_names"] == ["tag_1", "tag_2"]
    assert ckpt["class_names_per_factor"] == [["red", "yellow"], ["blue", "green"]]
    batch = torch.zeros(1, 3, 32, 32)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (1, 4)  # 2 + 2 concatenated

    shutil.rmtree(run_dir, ignore_errors=True)
