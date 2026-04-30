"""Verify TrackerKit consumer metadata-summary works on shared-trunk checkpoints.

This avoids spinning up Qt — we exercise the metadata-derivation paths that
the dialog and panel rely on.
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(
    os.environ.get("HYDRA_OFFLINE_TESTS") == "1",
    reason="needs torchvision pretrained weights",
)
def test_import_dialog_metadata_payload_shape(tmp_path):
    from hydra_suite.core.identity.classification.backend import ClassifierBackend
    from hydra_suite.training.multihead_torchvision_model import (
        build_multihead_torchvision_classifier,
    )
    from hydra_suite.training.torchvision_model import save_torchvision_checkpoint

    cnpf = [["red", "yellow"], ["blue", "green"]]
    model = build_multihead_torchvision_classifier(
        backbone="resnet18",
        class_names_per_factor=cnpf,
        trainable_layers=-1,
        head_hidden_dim=16,
        head_dropout=0.0,
        input_size=64,
    )
    p = tmp_path / "ckpt.pth"
    save_torchvision_checkpoint(
        model=model,
        backbone="resnet18",
        class_names=[],
        factor_names=["tag_1", "tag_2"],
        class_names_per_factor=cnpf,
        input_size=(64, 64),
        best_val_acc=0.5,
        history={"train_loss": [], "val_acc": []},
        trainable_layers=-1,
        backbone_lr_scale=0.1,
        monochrome=False,
        extra_meta={
            "head_kind": "multihead_shared_trunk",
            "head_hidden_dim": 16,
            "head_dropout": 0.0,
        },
        path=p,
    )

    backend = ClassifierBackend(str(p), compute_runtime="cpu")
    meta = backend.metadata
    summary = {
        "is_multihead": meta.is_multihead,
        "factor_names": list(meta.factor_names),
        "class_names_per_factor": [list(c) for c in meta.class_names_per_factor],
        "arch": meta.arch,
        "input_size": list(meta.input_size),
    }
    backend.close()

    assert summary["is_multihead"] is True
    assert summary["factor_names"] == ["tag_1", "tag_2"]
    assert summary["class_names_per_factor"] == cnpf
    assert summary["arch"] == "resnet18"
    assert summary["input_size"] == [64, 64]


@pytest.mark.skipif(
    os.environ.get("HYDRA_OFFLINE_TESTS") == "1",
    reason="needs torchvision pretrained weights",
)
def test_headtail_rejects_shared_trunk_classifier(tmp_path):
    from hydra_suite.training.multihead_torchvision_model import (
        build_multihead_torchvision_classifier,
    )
    from hydra_suite.training.torchvision_model import save_torchvision_checkpoint

    cnpf = [["red", "yellow"], ["blue", "green"]]
    model = build_multihead_torchvision_classifier(
        backbone="resnet18",
        class_names_per_factor=cnpf,
        trainable_layers=-1,
        head_hidden_dim=16,
        head_dropout=0.0,
        input_size=64,
    )
    p = tmp_path / "ckpt.pth"
    save_torchvision_checkpoint(
        model=model,
        backbone="resnet18",
        class_names=[],
        factor_names=["tag_1", "tag_2"],
        class_names_per_factor=cnpf,
        input_size=(64, 64),
        best_val_acc=0.0,
        history={"train_loss": [], "val_acc": []},
        trainable_layers=-1,
        backbone_lr_scale=0.1,
        monochrome=False,
        extra_meta={
            "head_kind": "multihead_shared_trunk",
            "head_hidden_dim": 16,
            "head_dropout": 0.0,
        },
        path=p,
    )

    # HeadTailAnalyzer requires a flat metadata; instantiating it on a
    # multi-head checkpoint must raise (preserving the legacy invariant).
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    try:
        HeadTailAnalyzer(model_path=str(p), compute_runtime="cpu")
        raise AssertionError("HeadTailAnalyzer accepted a multi-head classifier")
    except AssertionError:
        raise
    except Exception as exc:
        msg = str(exc).lower()
        assert "multi-head" in msg or "head-tail" in msg or "multihead" in msg
