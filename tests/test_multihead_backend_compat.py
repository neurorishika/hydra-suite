"""ClassifierBackend must load and predict on shared-trunk torchvision .pth."""

from __future__ import annotations

import os

import numpy as np
import pytest

from hydra_suite.runtime.resolver import ResolvedBackend


@pytest.mark.skipif(
    os.environ.get("HYDRA_OFFLINE_TESTS") == "1",
    reason="needs torchvision pretrained weights",
)
def test_backend_metadata_and_predict_for_shared_trunk(tmp_path):
    from hydra_suite.core.identity.classification.backend import ClassifierBackend
    from hydra_suite.training.multihead_torchvision_model import (
        build_multihead_torchvision_classifier,
    )
    from hydra_suite.training.torchvision_model import save_torchvision_checkpoint

    cnpf = [["a", "b"], ["x", "y", "z"]]
    model = build_multihead_torchvision_classifier(
        backbone="resnet18",
        class_names_per_factor=cnpf,
        trainable_layers=-1,
        head_hidden_dim=32,
        head_dropout=0.0,
        input_size=64,
    )
    ckpt_path = tmp_path / "shared_trunk.pth"
    save_torchvision_checkpoint(
        model=model,
        backbone="resnet18",
        class_names=[],
        factor_names=["color1", "color2"],
        class_names_per_factor=cnpf,
        input_size=(64, 64),
        best_val_acc=0.5,
        history={"train_loss": [], "val_acc": []},
        trainable_layers=-1,
        backbone_lr_scale=0.1,
        monochrome=False,
        extra_meta={
            "head_kind": "multihead_shared_trunk",
            "head_hidden_dim": 32,
            "head_dropout": 0.0,
        },
        path=ckpt_path,
    )

    backend = ClassifierBackend(
        str(ckpt_path), resolved=ResolvedBackend("torch", "cpu", False)
    )
    meta = backend.metadata
    assert meta.is_multihead is True
    assert meta.factor_names == ["color1", "color2"]
    assert meta.class_names_per_factor == cnpf
    assert meta.arch == "resnet18"  # NOT "classifier_multihead"

    crops = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]
    per_crop = backend.predict_batch(crops)
    assert len(per_crop) == 3
    for factors in per_crop:
        assert len(factors) == 2  # K factors
        assert factors[0].shape == (2,)
        assert factors[1].shape == (3,)
        # Probability simplex per factor
        assert abs(float(factors[0].sum()) - 1.0) < 1e-4
        assert abs(float(factors[1].sum()) - 1.0) < 1e-4
    backend.close()
