"""Tests for ClassKit extended training — Custom CNN models."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest
import torch


def test_custom_cnn_params_defaults():
    from hydra_suite.training.contracts import CustomCNNParams

    p = CustomCNNParams()
    assert p.backbone == "tinyclassifier"
    assert p.fine_tune_method == "head_only"
    assert p.trainable_layers == 0
    assert p.backbone_lr_scale == 0.1
    assert p.layerwise_lr_decay == 0.75
    assert p.gradual_unfreeze_interval == 5
    assert p.input_size == 224
    assert p.epochs == 50
    assert p.batch == 32
    assert p.lr == 1e-3
    assert p.weight_decay == 1e-2
    assert p.patience == 10
    assert p.label_smoothing == 0.0
    assert p.class_rebalance_mode == "none"
    assert p.class_rebalance_power == 1.0
    assert p.hidden_layers == 1
    assert p.hidden_dim == 96
    assert p.dropout == 0.1
    assert p.input_width == 128
    assert p.input_height == 64


def test_new_training_roles_exist():
    from hydra_suite.training.contracts import TrainingRole

    assert TrainingRole.CLASSIFY_FLAT_CUSTOM.value == "classify_flat_custom"
    assert TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM.value == "classify_multihead_custom"


def test_training_run_spec_has_custom_params():
    from hydra_suite.training.contracts import (
        CustomCNNParams,
        TrainingRole,
        TrainingRunSpec,
    )

    spec = TrainingRunSpec(
        role=TrainingRole.CLASSIFY_FLAT_CUSTOM,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="",
        hyperparams=None,
    )
    assert hasattr(spec, "custom_params")
    assert spec.custom_params is None

    spec2 = TrainingRunSpec(
        role=TrainingRole.CLASSIFY_FLAT_CUSTOM,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="",
        hyperparams=None,
        custom_params=CustomCNNParams(
            backbone="convnext_tiny", fine_tune_method="full_finetune"
        ),
    )
    assert spec2.custom_params.backbone == "convnext_tiny"
    assert spec2.custom_params.fine_tune_method == "full_finetune"


def test_custom_backbone_registry_persists(monkeypatch, tmp_path):
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(tmp_path / "cfg"))

    from hydra_suite.classkit.config.custom_backbones import (
        get_custom_backbone_choices,
        load_user_timm_backbones,
        register_user_timm_backbones,
    )

    register_user_timm_backbones(
        ["timm/resnet50.a1_in1k", "timm/convnext_tiny.fb_in1k"]
    )

    loaded = load_user_timm_backbones()
    assert loaded == ["timm/resnet50.a1_in1k", "timm/convnext_tiny.fb_in1k"]
    choices = get_custom_backbone_choices()
    assert "tinyclassifier" in choices
    assert "mobilenet_v3_small" in choices
    assert "shufflenet_v2_x1_0" in choices
    assert "timm/resnet50.a1_in1k" in choices


# ---------------------------------------------------------------------------
# torchvision_model tests
# ---------------------------------------------------------------------------


def test_build_torchvision_classifier_convnext():
    from hydra_suite.training.torchvision_model import build_torchvision_classifier

    model = build_torchvision_classifier(
        "convnext_tiny", num_classes=5, trainable_layers=0
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 5)


def test_build_torchvision_classifier_efficientnet():
    from hydra_suite.training.torchvision_model import build_torchvision_classifier

    model = build_torchvision_classifier(
        "efficientnet_b0", num_classes=3, trainable_layers=0
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 3)


def test_build_torchvision_classifier_mobilenet():
    from hydra_suite.training.torchvision_model import build_torchvision_classifier

    model = build_torchvision_classifier(
        "mobilenet_v3_small", num_classes=3, trainable_layers=0
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 3)


def test_build_torchvision_classifier_resnet():
    from hydra_suite.training.torchvision_model import build_torchvision_classifier

    model = build_torchvision_classifier("resnet18", num_classes=4, trainable_layers=0)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 4)


def test_build_torchvision_classifier_shufflenet():
    from hydra_suite.training.torchvision_model import build_torchvision_classifier

    model = build_torchvision_classifier(
        "shufflenet_v2_x1_0", num_classes=4, trainable_layers=0
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 4)


def test_build_torchvision_classifier_vit():
    from hydra_suite.training.torchvision_model import build_torchvision_classifier

    model = build_torchvision_classifier("vit_b_16", num_classes=6, trainable_layers=0)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 6)


def test_get_layer_groups_convnext_count():
    from hydra_suite.training.torchvision_model import (
        build_torchvision_classifier,
        get_layer_groups,
    )

    model = build_torchvision_classifier(
        "convnext_tiny", num_classes=2, trainable_layers=0
    )
    groups = get_layer_groups(model, "convnext_tiny")
    assert len(groups) == 4


def test_get_layer_groups_resnet_count():
    from hydra_suite.training.torchvision_model import (
        build_torchvision_classifier,
        get_layer_groups,
    )

    model = build_torchvision_classifier("resnet18", num_classes=2, trainable_layers=0)
    groups = get_layer_groups(model, "resnet18")
    assert len(groups) == 4


def test_get_layer_groups_mobilenet_count():
    from hydra_suite.training.torchvision_model import (
        build_torchvision_classifier,
        get_layer_groups,
    )

    model = build_torchvision_classifier(
        "mobilenet_v3_small", num_classes=2, trainable_layers=0
    )
    groups = get_layer_groups(model, "mobilenet_v3_small")
    assert len(groups) == len(model.features)


def test_get_layer_groups_efficientnet_count():
    from hydra_suite.training.torchvision_model import (
        build_torchvision_classifier,
        get_layer_groups,
    )

    model = build_torchvision_classifier(
        "efficientnet_b0", num_classes=2, trainable_layers=0
    )
    groups = get_layer_groups(model, "efficientnet_b0")
    # EfficientNet-B0 features Sequential has 9 blocks (indices 0–8)
    assert len(groups) == 9


def test_get_layer_groups_shufflenet_count():
    from hydra_suite.training.torchvision_model import (
        build_torchvision_classifier,
        get_layer_groups,
    )

    model = build_torchvision_classifier(
        "shufflenet_v2_x1_0", num_classes=2, trainable_layers=0
    )
    groups = get_layer_groups(model, "shufflenet_v2_x1_0")
    assert len(groups) == 4


def test_get_layer_groups_vit_count():
    from hydra_suite.training.torchvision_model import (
        build_torchvision_classifier,
        get_layer_groups,
    )

    model = build_torchvision_classifier("vit_b_16", num_classes=2, trainable_layers=0)
    groups = get_layer_groups(model, "vit_b_16")
    # ViT-B/16 encoder has 12 transformer layers
    assert len(groups) == 12


def test_freeze_backbone_frozen():
    from hydra_suite.training.torchvision_model import build_torchvision_classifier

    model = build_torchvision_classifier("resnet18", num_classes=2, trainable_layers=0)
    # Head (fc) must be unfrozen
    assert model.fc.weight.requires_grad is True
    # At least some backbone parameters must be frozen
    backbone_frozen = any(
        not p.requires_grad for name, p in model.named_parameters() if "fc" not in name
    )
    assert backbone_frozen


def test_freeze_backbone_all():
    from hydra_suite.training.torchvision_model import build_torchvision_classifier

    model = build_torchvision_classifier("resnet18", num_classes=2, trainable_layers=-1)
    # All parameters must be trainable
    assert all(p.requires_grad for p in model.parameters())


def test_freeze_backbone_partial():
    from hydra_suite.training.torchvision_model import (
        build_torchvision_classifier,
        get_layer_groups,
    )

    model = build_torchvision_classifier("resnet18", num_classes=2, trainable_layers=1)
    groups = get_layer_groups(model, "resnet18")
    # Last group (layer4) must be unfrozen
    assert any(p.requires_grad for p in groups[-1].parameters())
    # First group (layer1) must be frozen
    assert all(not p.requires_grad for p in groups[0].parameters())


def test_checkpoint_format_required_keys(tmp_path):
    from hydra_suite.training.torchvision_model import (
        build_torchvision_classifier,
        save_torchvision_checkpoint,
    )

    model = build_torchvision_classifier("resnet18", num_classes=2, trainable_layers=0)
    ckpt_path = tmp_path / "test.pth"
    save_torchvision_checkpoint(
        model=model,
        backbone="resnet18",
        class_names=["a", "b"],
        factor_names=[],
        input_size=(224, 224),
        best_val_acc=0.95,
        history={"train_loss": [], "val_acc": []},
        trainable_layers=0,
        backbone_lr_scale=0.1,
        path=ckpt_path,
    )
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    required = {
        "arch",
        "class_names",
        "factor_names",
        "input_size",
        "num_classes",
        "model_state_dict",
        "best_val_acc",
        "history",
        "trainable_layers",
        "backbone_lr_scale",
    }
    assert required.issubset(set(ckpt.keys()))


def test_load_torchvision_classifier_roundtrip(tmp_path):
    from hydra_suite.training.torchvision_model import (
        build_torchvision_classifier,
        load_torchvision_classifier,
        save_torchvision_checkpoint,
    )

    model = build_torchvision_classifier("resnet18", num_classes=3, trainable_layers=0)
    model.eval()
    ckpt_path = tmp_path / "model.pth"
    save_torchvision_checkpoint(
        model=model,
        backbone="resnet18",
        class_names=["x", "y", "z"],
        factor_names=[],
        input_size=(224, 224),
        best_val_acc=0.9,
        history={},
        trainable_layers=0,
        backbone_lr_scale=0.1,
        path=ckpt_path,
    )
    loaded_model, ckpt = load_torchvision_classifier(str(ckpt_path), device="cpu")
    loaded_model.eval()
    x = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        orig_out = model(x)
        loaded_out = loaded_model(x)
    assert torch.allclose(orig_out, loaded_out, atol=1e-5)
    assert ckpt["arch"] == "resnet18"
    assert ckpt["class_names"] == ["x", "y", "z"]


def test_export_torchvision_to_onnx_smoke(tmp_path):
    from hydra_suite.training.torchvision_model import (
        build_torchvision_classifier,
        export_torchvision_to_onnx,
        save_torchvision_checkpoint,
    )

    model = build_torchvision_classifier("resnet18", num_classes=2, trainable_layers=0)
    ckpt_path = tmp_path / "model.pth"
    save_torchvision_checkpoint(
        model=model,
        backbone="resnet18",
        class_names=["a", "b"],
        factor_names=[],
        input_size=(224, 224),
        best_val_acc=0.8,
        history={},
        trainable_layers=0,
        backbone_lr_scale=0.1,
        path=ckpt_path,
    )
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    onnx_path = tmp_path / "model.onnx"
    export_torchvision_to_onnx(model, ckpt, onnx_path)
    assert onnx_path.exists()
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(
        None, {sess.get_inputs()[0].name: np.zeros((1, 3, 224, 224), dtype=np.float32)}
    )
    assert out[0].shape == (1, 2)


# ---------------------------------------------------------------------------
# Spec-required tests: backbone registry and tinyclassifier integration
# ---------------------------------------------------------------------------


def test_torchvision_backbones_list_contains_expected():
    from hydra_suite.training.torchvision_model import TORCHVISION_BACKBONES

    assert "convnext_tiny" in TORCHVISION_BACKBONES
    assert "mobilenet_v3_small" in TORCHVISION_BACKBONES
    assert "shufflenet_v2_x1_0" in TORCHVISION_BACKBONES
    assert "tinyclassifier" in TORCHVISION_BACKBONES
    assert "vit_b_16" in TORCHVISION_BACKBONES


def test_backbone_display_names_covers_all_backbones():
    from hydra_suite.training.torchvision_model import (
        BACKBONE_DISPLAY_NAMES,
        TORCHVISION_BACKBONES,
    )

    for b in TORCHVISION_BACKBONES:
        assert b in BACKBONE_DISPLAY_NAMES


def test_build_torchvision_classifier_raises_for_unknown_backbone():
    import pytest

    from hydra_suite.training.torchvision_model import build_torchvision_classifier

    with pytest.raises(ValueError, match="Unknown backbone"):
        build_torchvision_classifier(
            "not_a_real_backbone", num_classes=2, trainable_layers=0
        )


def test_build_torchvision_classifier_tinyclassifier():
    from hydra_suite.training.contracts import CustomCNNParams
    from hydra_suite.training.torchvision_model import build_torchvision_classifier

    params = (
        CustomCNNParams()
    )  # defaults: hidden_layers=1, hidden_dim=96, dropout=0.1, input_width=128, input_height=64
    model = build_torchvision_classifier(
        "tinyclassifier",
        num_classes=3,
        trainable_layers=0,
        tiny_preset=params.tiny_preset,
        hidden_layers=params.hidden_layers,
        hidden_dim=params.hidden_dim,
        dropout=params.dropout,
        input_width=params.input_width,
        input_height=params.input_height,
    )
    assert model is not None
    x = torch.randn(2, 3, params.input_height, params.input_width)
    out = model(x)
    assert out.shape == (2, 3)


def test_build_torchvision_classifier_tinyclassifier_large_preset():
    from hydra_suite.training.torchvision_model import build_torchvision_classifier

    model = build_torchvision_classifier(
        "tinyclassifier",
        num_classes=2,
        trainable_layers=0,
        tiny_preset="large",
    )

    assert getattr(model, "tiny_preset", None) == "large"
    assert tuple(getattr(model, "feature_channels", ())) == (32, 64, 128, 224)


# ---------------------------------------------------------------------------
# Task 3: runner dispatch tests
# ---------------------------------------------------------------------------


def test_runner_flat_tiny_alias_dispatches_to_custom_classify():
    """flat_tiny role should call _train_custom_classify with backbone='tinyclassifier'."""
    from hydra_suite.training import runner
    from hydra_suite.training.contracts import (
        CustomCNNParams,
        TrainingRole,
        TrainingRunSpec,
    )

    spec = TrainingRunSpec(
        role=TrainingRole.CLASSIFY_FLAT_TINY,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="",
        hyperparams=None,
        custom_params=CustomCNNParams(backbone="tinyclassifier"),
    )
    with patch.object(
        runner, "_train_custom_classify", return_value={"success": True}
    ) as mock_fn:
        runner.run_training(spec, "/tmp/run")
        mock_fn.assert_called_once()
        call_spec = mock_fn.call_args[0][0]
        assert call_spec.custom_params.backbone == "tinyclassifier"


def test_runner_flat_custom_dispatches_to_custom_classify():
    """flat_custom role should call _train_custom_classify."""
    from hydra_suite.training import runner
    from hydra_suite.training.contracts import (
        CustomCNNParams,
        TrainingRole,
        TrainingRunSpec,
    )

    spec = TrainingRunSpec(
        role=TrainingRole.CLASSIFY_FLAT_CUSTOM,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="",
        hyperparams=None,
        custom_params=CustomCNNParams(backbone="resnet18"),
    )
    with patch.object(
        runner, "_train_custom_classify", return_value={"success": True}
    ) as mock_fn:
        runner.run_training(spec, "/tmp/run")
        mock_fn.assert_called_once()


def test_train_custom_tinyclassifier_translates_custom_params_to_tiny_params():
    sys.modules.setdefault("cv2", ModuleType("cv2"))

    from hydra_suite.training import runner
    from hydra_suite.training.contracts import (
        AugmentationProfile,
        CustomCNNParams,
        TinyHeadTailParams,
        TrainingHyperParams,
        TrainingRole,
        TrainingRunSpec,
    )

    spec = TrainingRunSpec(
        role=TrainingRole.CLASSIFY_FLAT_CUSTOM,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="",
        hyperparams=TrainingHyperParams(),
        augmentation_profile=AugmentationProfile(monochrome=True),
        tiny_params=TinyHeadTailParams(),
        custom_params=CustomCNNParams(
            backbone="tinyclassifier",
            epochs=7,
            batch=9,
            lr=0.004,
            weight_decay=0.03,
            patience=4,
            hidden_layers=2,
            hidden_dim=96,
            dropout=0.05,
            input_width=80,
            input_height=72,
            class_rebalance_mode="weighted_loss",
            class_rebalance_power=1.5,
            label_smoothing=0.08,
        ),
    )

    with patch.object(
        runner, "_train_tiny_classify", return_value={"success": True}
    ) as mock_fn:
        runner._train_custom_classify(spec, Path("/tmp/run"))

    translated_spec = mock_fn.call_args[0][0]
    assert translated_spec is not spec
    assert translated_spec.augmentation_profile.monochrome is True
    assert translated_spec.tiny_params.epochs == 7
    assert translated_spec.tiny_params.batch == 9
    assert translated_spec.tiny_params.lr == pytest.approx(0.004)
    assert translated_spec.tiny_params.weight_decay == pytest.approx(0.03)
    assert translated_spec.tiny_params.patience == 4
    assert translated_spec.tiny_params.hidden_layers == 2
    assert translated_spec.tiny_params.hidden_dim == 96
    assert translated_spec.tiny_params.dropout == pytest.approx(0.05)
    assert translated_spec.tiny_params.input_width == 80
    assert translated_spec.tiny_params.input_height == 72
    assert translated_spec.tiny_params.class_rebalance_mode == "weighted_loss"
    assert translated_spec.tiny_params.class_rebalance_power == pytest.approx(1.5)
    assert translated_spec.tiny_params.label_smoothing == pytest.approx(0.08)


def test_train_custom_classify_remaps_validation_targets_to_shared_class_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from hydra_suite.training import runner, torchvision_model
    from hydra_suite.training.contracts import (
        AugmentationProfile,
        CustomCNNParams,
        TrainingHyperParams,
        TrainingRole,
        TrainingRunSpec,
    )

    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "train" / "ant").mkdir(parents=True)
    (dataset_dir / "train" / "bee").mkdir(parents=True)
    (dataset_dir / "val" / "bee").mkdir(parents=True)
    (dataset_dir / "train" / "ant" / "ant_0.png").write_bytes(b"train-ant")
    (dataset_dir / "train" / "bee" / "bee_0.png").write_bytes(b"train-bee")
    (dataset_dir / "val" / "bee" / "bee_1.png").write_bytes(b"val-bee")

    class FakeImageFolder:
        def __init__(self, root, transform=None):
            self.root = str(root)
            self.transform = transform
            if str(root).endswith("/train"):
                self.class_to_idx = {"ant": 0, "bee": 1}
                self.classes = ["ant", "bee"]
                self.samples = [
                    (str(Path(root) / "ant" / "ant_0.png"), 0),
                    (str(Path(root) / "bee" / "bee_0.png"), 1),
                ]
            else:
                self.class_to_idx = {"bee": 0}
                self.classes = ["bee"]
                self.samples = [(str(Path(root) / "bee" / "bee_1.png"), 0)]
            self.targets = [label for _, label in self.samples]
            self.imgs = list(self.samples)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return torch.zeros(3, 64, 64), self.targets[index]

    spec = TrainingRunSpec(
        role=TrainingRole.CLASSIFY_FLAT_CUSTOM,
        source_datasets=[],
        derived_dataset_dir=str(dataset_dir),
        base_model="",
        hyperparams=TrainingHyperParams(),
        device="cpu",
        augmentation_profile=AugmentationProfile(enabled=False),
        custom_params=CustomCNNParams(
            backbone="resnet18",
            input_size=64,
            epochs=1,
            batch=2,
            patience=1,
        ),
    )

    def _fake_training_loop(
        model,
        backbone,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        params,
        save_torchvision_checkpoint,
        freeze_backbone,
        get_layer_groups,
        best_ckpt_path,
        class_names,
        checkpoint_extra_meta,
        log_cb,
        progress_cb,
        should_cancel,
    ):
        assert train_loader.dataset.class_to_idx == {"ant": 0, "bee": 1}
        assert train_loader.dataset.targets == [0, 1]
        assert val_loader is not None
        assert val_loader.dataset.class_to_idx == {"ant": 0, "bee": 1}
        assert val_loader.dataset.classes == ["ant", "bee"]
        assert val_loader.dataset.targets == [1]
        assert val_loader.dataset.samples[0][1] == 1
        best_ckpt_path.write_bytes(b"stub")
        return 1.0, {"train_loss": [0.1], "val_acc": [1.0]}

    monkeypatch.setattr("torchvision.datasets.ImageFolder", FakeImageFolder)
    monkeypatch.setattr(
        runner, "_build_class_to_idx", lambda _path: {"ant": 0, "bee": 1}
    )
    monkeypatch.setattr(
        torchvision_model,
        "build_torchvision_classifier",
        lambda *args, **kwargs: torch.nn.Linear(4, 2),
    )
    monkeypatch.setattr(
        torchvision_model, "freeze_backbone", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        torchvision_model, "get_layer_groups", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        runner,
        "_build_optimizer_for_fine_tune_strategy",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(runner, "_pick_torch_device", lambda _device: "cpu")
    monkeypatch.setattr(runner, "_run_torchvision_training_loop", _fake_training_loop)
    monkeypatch.setattr(
        torchvision_model,
        "load_torchvision_classifier",
        lambda *args, **kwargs: (
            torch.nn.Linear(4, 2),
            {"arch": "resnet18", "class_names": ["ant", "bee"], "input_size": (64, 64)},
        ),
    )
    monkeypatch.setattr(
        torchvision_model,
        "export_torchvision_to_onnx",
        lambda model, ckpt, path: Path(path).write_bytes(b"onnx"),
    )

    result = runner._train_custom_classify(spec, tmp_path / "run")

    assert result["success"] is True
    assert Path(result["artifact_path"]).exists()


def test_load_compatible_checkpoint_weights_skips_mismatched_head(tmp_path: Path):
    from hydra_suite.training.runner import _load_compatible_checkpoint_weights

    class ToyClassifier(torch.nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.features = torch.nn.Linear(4, 4)
            self.classifier = torch.nn.Linear(4, num_classes)

    source = ToyClassifier(num_classes=2)
    target = ToyClassifier(num_classes=3)

    checkpoint_path = tmp_path / "toy.pth"
    torch.save(
        {
            "arch": "toy_backbone",
            "model_state_dict": source.state_dict(),
        },
        checkpoint_path,
    )

    original_target_head = target.classifier.weight.detach().clone()
    result = _load_compatible_checkpoint_weights(
        target,
        checkpoint_path,
        expected_arch="toy_backbone",
    )

    assert result["loaded"] == 2
    assert result["skipped"] == 2
    assert torch.equal(target.features.weight, source.features.weight)
    assert not torch.equal(target.classifier.weight, source.classifier.weight)
    assert torch.equal(target.classifier.weight, original_target_head)


def test_load_compatible_checkpoint_weights_rejects_arch_mismatch(tmp_path: Path):
    from hydra_suite.training.runner import _load_compatible_checkpoint_weights

    model = torch.nn.Linear(4, 2)
    checkpoint_path = tmp_path / "toy_bad_arch.pth"
    torch.save(
        {
            "arch": "other_backbone",
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="backbone"):
        _load_compatible_checkpoint_weights(
            model,
            checkpoint_path,
            expected_arch="toy_backbone",
        )


def test_all_presets_include_flat_custom():
    """Every single-factor preset function must include 'flat_custom' in training_modes."""
    from hydra_suite.classkit.config.presets import (
        age_preset,
        apriltag_preset,
        head_tail_preset,
    )

    assert "flat_custom" in apriltag_preset("tag36h11", 9).training_modes
    assert "flat_custom" in head_tail_preset().training_modes
    assert "flat_custom" in age_preset().training_modes


def test_color_tag_preset_includes_multihead_custom():
    """Multi-factor preset must include both flat_custom and multihead_custom."""
    from hydra_suite.classkit.config.presets import color_tag_preset

    scheme = color_tag_preset(n_factors=2, colors=["red", "blue"])
    assert "flat_custom" in scheme.training_modes
    assert "multihead_custom" in scheme.training_modes
