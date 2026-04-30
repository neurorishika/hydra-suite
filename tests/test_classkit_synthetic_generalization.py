from __future__ import annotations

import random
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

pytest.importorskip("torchvision")


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _pattern_seed(split: str, class_name: str, index: int) -> int:
    split_offsets = {"train": 11, "val": 23, "test": 37}
    class_offsets = {"vertical": 101, "horizontal": 211}
    return split_offsets[split] * 10_000 + class_offsets[class_name] + index


def _make_pattern_image(
    class_name: str, *, split: str, index: int, size: int
) -> Image.Image:
    rng = np.random.default_rng(_pattern_seed(split, class_name, index))
    image = rng.integers(0, 20, size=(size, size, 3), dtype=np.uint8)

    image = np.clip(
        image.astype(np.int16) + int(rng.integers(0, 18)),
        0,
        255,
    ).astype(np.uint8)

    thickness = int(rng.integers(max(4, size // 10), max(6, size // 6)))
    accent = int(rng.integers(175, 245))

    if class_name == "vertical":
        center_x = int(rng.integers(size // 4, (3 * size) // 4))
        left = max(0, center_x - thickness // 2)
        right = min(size, center_x + thickness // 2 + 1)
        image[:, left:right, 0] = accent
        image[:, left:right, 1] = np.clip(image[:, left:right, 1] + 35, 0, 255)
    elif class_name == "horizontal":
        center_y = int(rng.integers(size // 4, (3 * size) // 4))
        top = max(0, center_y - thickness // 2)
        bottom = min(size, center_y + thickness // 2 + 1)
        image[top:bottom, :, 1] = accent
        image[top:bottom, :, 2] = np.clip(image[top:bottom, :, 2] + 35, 0, 255)
    else:
        raise ValueError(f"Unsupported synthetic class: {class_name}")

    # Add one distractor patch so the model must learn the dominant pattern.
    patch_size = max(3, size // 12)
    patch_x = int(rng.integers(0, size - patch_size + 1))
    patch_y = int(rng.integers(0, size - patch_size + 1))
    image[patch_y : patch_y + patch_size, patch_x : patch_x + patch_size, :] = (
        rng.integers(
            0,
            100,
            size=(patch_size, patch_size, 3),
            dtype=np.uint8,
        )
    )

    return Image.fromarray(image, mode="RGB")


def _build_synthetic_classify_dataset(
    root: Path,
    *,
    image_size: int = 64,
    split_sizes: dict[str, int] | None = None,
) -> Path:
    split_sizes = split_sizes or {"train": 18, "val": 8, "test": 8}
    classes = ["vertical", "horizontal"]
    for split, count in split_sizes.items():
        for class_name in classes:
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for index in range(int(count)):
                image = _make_pattern_image(
                    class_name,
                    split=split,
                    index=index,
                    size=image_size,
                )
                image.save(class_dir / f"{class_name}_{index:03d}.png")
    return root


def _make_multifactor_pattern_image(
    orientation: str,
    color_name: str,
    *,
    split: str,
    index: int,
    size: int,
) -> Image.Image:
    color_offsets = {"red": 401, "green": 503}
    combined_index = index + color_offsets[color_name]
    image = np.asarray(
        _make_pattern_image(orientation, split=split, index=combined_index, size=size),
        dtype=np.uint8,
    ).copy()

    if color_name == "red":
        image[..., 0] = np.clip(image[..., 0].astype(np.int16) + 60, 0, 255).astype(
            np.uint8
        )
    elif color_name == "green":
        image[..., 1] = np.clip(image[..., 1].astype(np.int16) + 60, 0, 255).astype(
            np.uint8
        )
    else:
        raise ValueError(f"Unsupported synthetic color factor: {color_name}")

    return Image.fromarray(image, mode="RGB")


def _build_multifactor_samples(
    root: Path,
    *,
    image_size: int = 64,
    split_sizes: dict[str, int] | None = None,
) -> list[dict[str, object]]:
    split_sizes = split_sizes or {"train": 10, "val": 4, "test": 4}
    samples: list[dict[str, object]] = []
    for split, count in split_sizes.items():
        for orientation in ("vertical", "horizontal"):
            for color_name in ("red", "green"):
                for index in range(int(count)):
                    image = _make_multifactor_pattern_image(
                        orientation,
                        color_name,
                        split=split,
                        index=index,
                        size=image_size,
                    )
                    out_path = (
                        root / split / f"{orientation}_{color_name}_{index:03d}.png"
                    )
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(out_path)
                    samples.append(
                        {
                            "path": out_path,
                            "split": split,
                            "orientation": orientation,
                            "color": color_name,
                            "composite": f"{orientation}|{color_name}",
                        }
                    )
    return samples


def _evaluate_tiny_checkpoint(artifact_path: Path, dataset_dir: Path) -> float:
    from hydra_suite.training.runner import _build_class_to_idx, _iter_classify_samples
    from hydra_suite.training.tiny_model import load_tiny_classifier

    model, ckpt = load_tiny_classifier(artifact_path, device="cpu")
    input_w, input_h = ckpt.get("input_size", [128, 64])
    force_monochrome = bool(ckpt.get("monochrome", False))
    class_to_idx = _build_class_to_idx(dataset_dir)
    samples = list(_iter_classify_samples(dataset_dir, "test", class_to_idx))

    correct = 0
    with torch.no_grad():
        for image_path, label in samples:
            image = Image.open(image_path).convert("RGB").resize((input_w, input_h))
            if force_monochrome:
                image = image.convert("L").convert("RGB")
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
            prediction = model(tensor).argmax(dim=1).item()
            correct += int(prediction == int(label))
    return correct / max(len(samples), 1)


def _evaluate_torchvision_checkpoint(artifact_path: Path, dataset_dir: Path) -> float:
    from torchvision import datasets, transforms

    from hydra_suite.training.torchvision_model import (
        get_classifier_normalization_stats,
        load_torchvision_classifier,
    )

    model, ckpt = load_torchvision_classifier(artifact_path, device="cpu")
    input_h, input_w = ckpt.get("input_size", (224, 224))
    mean, std = get_classifier_normalization_stats(
        monochrome=bool(ckpt.get("monochrome", False))
    )
    transform_steps = [transforms.Resize((input_h, input_w))]
    if bool(ckpt.get("monochrome", False)):
        transform_steps.append(transforms.Grayscale(num_output_channels=3))
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    dataset = datasets.ImageFolder(
        str(dataset_dir / "test"),
        transform=transforms.Compose(transform_steps),
    )
    assert dataset.classes == list(ckpt["class_names"])

    correct = 0
    with torch.no_grad():
        for tensor, label in dataset:
            prediction = model(tensor.unsqueeze(0)).argmax(dim=1).item()
            correct += int(prediction == int(label))
    return correct / max(len(dataset), 1)


def _evaluate_artifact_accuracy(artifact_path: Path, dataset_dir: Path) -> float:
    ckpt = torch.load(str(artifact_path), map_location="cpu", weights_only=False)
    if ckpt.get("arch") == "tinyclassifier":
        return _evaluate_tiny_checkpoint(artifact_path, dataset_dir)
    return _evaluate_torchvision_checkpoint(artifact_path, dataset_dir)


def _build_pil_tiny_dataset_class(input_w: int, input_h: int):
    class TinyDataset(Dataset):
        def __init__(self, items, augment: bool = False, profile=None):
            self.items = list(items)
            self.augment = bool(augment)
            self.profile = profile

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            path, label = self.items[idx]
            image = Image.open(path).convert("RGB")
            if getattr(self.profile, "monochrome", False):
                image = image.convert("L").convert("RGB")
            image = image.resize((input_w, input_h))
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
            return tensor, torch.tensor(label, dtype=torch.long)

    return TinyDataset


def _patch_tiny_dataset_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "hydra_suite.training.runner._build_tiny_dataset_class",
        _build_pil_tiny_dataset_class,
    )


def _ensure_stub_cv2_module(monkeypatch: pytest.MonkeyPatch) -> None:
    if "cv2" in sys.modules:
        return
    monkeypatch.setitem(sys.modules, "cv2", ModuleType("cv2"))


@pytest.mark.parametrize(
    ("case_name", "build_spec", "threshold"),
    [
        (
            "flat_tiny_rebalanced",
            lambda dataset_dir: _make_tiny_spec(dataset_dir),
            0.95,
        ),
        (
            "flat_custom_tiny_monochrome",
            lambda dataset_dir: _make_custom_tiny_spec(dataset_dir),
            0.95,
        ),
        (
            "flat_custom_resnet18_layerwise",
            lambda dataset_dir: _make_custom_resnet_spec(dataset_dir),
            0.85,
        ),
    ],
)
def test_classkit_trainer_generalizes_on_easy_synthetic_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
    build_spec: Callable[[Path], object],
    threshold: float,
) -> None:
    _ensure_stub_cv2_module(monkeypatch)

    from torchvision import models as tvm

    from hydra_suite.training.runner import run_training

    dataset_dir = _build_synthetic_classify_dataset(tmp_path / "dataset")
    run_dir = tmp_path / f"run_{case_name}"

    # Avoid network downloads when exercising the torchvision training path.
    monkeypatch.setattr(
        "hydra_suite.training.torchvision_model._load_pretrained",
        lambda backbone: getattr(tvm, backbone)(weights=None),
    )
    _patch_tiny_dataset_loader(monkeypatch)

    _set_global_seeds(1234)
    spec = build_spec(dataset_dir)
    result = run_training(spec, run_dir)

    assert result["success"] is True
    assert Path(result["artifact_path"]).exists()
    assert result.get("best_val_acc") is not None

    _set_global_seeds(1234)
    test_accuracy = _evaluate_artifact_accuracy(
        Path(result["artifact_path"]), dataset_dir
    )
    assert test_accuracy >= threshold, (
        f"{case_name} failed synthetic held-out generalization: "
        f"test_acc={test_accuracy:.3f}, expected>={threshold:.2f}"
    )


def _make_tiny_spec(dataset_dir: Path):
    from hydra_suite.training.contracts import (
        AugmentationProfile,
        TinyHeadTailParams,
        TrainingHyperParams,
        TrainingRole,
        TrainingRunSpec,
    )

    return TrainingRunSpec(
        role=TrainingRole.CLASSIFY_FLAT_TINY,
        source_datasets=[],
        derived_dataset_dir=str(dataset_dir),
        base_model="",
        hyperparams=TrainingHyperParams(),
        device="cpu",
        seed=1234,
        augmentation_profile=AugmentationProfile(enabled=False),
        tiny_params=TinyHeadTailParams(
            epochs=6,
            batch=8,
            lr=3e-3,
            patience=6,
            input_width=64,
            input_height=64,
            hidden_layers=1,
            hidden_dim=48,
            dropout=0.1,
            class_rebalance_mode="both",
            class_rebalance_power=1.0,
            label_smoothing=0.02,
        ),
    )


def _make_custom_tiny_spec(dataset_dir: Path):
    from hydra_suite.training.contracts import (
        AugmentationProfile,
        CustomCNNParams,
        TrainingHyperParams,
        TrainingRole,
        TrainingRunSpec,
    )

    return TrainingRunSpec(
        role=TrainingRole.CLASSIFY_FLAT_CUSTOM,
        source_datasets=[],
        derived_dataset_dir=str(dataset_dir),
        base_model="",
        hyperparams=TrainingHyperParams(),
        device="cpu",
        seed=1234,
        augmentation_profile=AugmentationProfile(enabled=True, monochrome=True),
        custom_params=CustomCNNParams(
            backbone="tinyclassifier",
            epochs=6,
            batch=8,
            lr=3e-3,
            patience=6,
            hidden_layers=1,
            hidden_dim=48,
            dropout=0.1,
            input_width=64,
            input_height=64,
            class_rebalance_mode="weighted_loss",
            class_rebalance_power=1.0,
            label_smoothing=0.02,
        ),
    )


def _make_custom_resnet_spec(dataset_dir: Path):
    from hydra_suite.training.contracts import (
        AugmentationProfile,
        CustomCNNParams,
        TrainingHyperParams,
        TrainingRole,
        TrainingRunSpec,
    )

    return TrainingRunSpec(
        role=TrainingRole.CLASSIFY_FLAT_CUSTOM,
        source_datasets=[],
        derived_dataset_dir=str(dataset_dir),
        base_model="",
        hyperparams=TrainingHyperParams(),
        device="cpu",
        seed=1234,
        augmentation_profile=AugmentationProfile(enabled=False),
        custom_params=CustomCNNParams(
            backbone="resnet18",
            fine_tune_method="layerwise_lr_decay",
            layerwise_lr_decay=0.85,
            input_size=64,
            epochs=6,
            batch=4,
            lr=1e-3,
            patience=6,
            label_smoothing=0.0,
        ),
    )


def _run_export_worker(worker) -> dict:
    payload: dict = {}
    errors: list[str] = []
    worker.signals.success.connect(payload.update)
    worker.signals.error.connect(errors.append)
    worker.run()
    assert not errors, errors[0] if errors else "export failed"
    return payload


def _evaluate_dataset_with_checkpoint(artifact_path: Path, dataset_dir: Path) -> float:
    return _evaluate_artifact_accuracy(artifact_path, dataset_dir)


@pytest.mark.parametrize(
    ("trainer_kind", "threshold"),
    [("tiny", 0.95), ("custom_tiny", 0.93)],
)
def test_classkit_multihead_factor_exports_share_splits_and_generalize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainer_kind: str,
    threshold: float,
) -> None:
    _ensure_stub_cv2_module(monkeypatch)

    from hydra_suite.classkit.jobs.task_workers import ExportWorker
    from hydra_suite.training.contracts import (
        AugmentationProfile,
        CustomCNNParams,
        TinyHeadTailParams,
        TrainingHyperParams,
        TrainingRole,
        TrainingRunSpec,
    )
    from hydra_suite.training.runner import run_training

    samples = _build_multifactor_samples(tmp_path / "multihead_source")
    image_paths = [Path(sample["path"]) for sample in samples]
    preset_splits = {
        str(Path(sample["path"]).resolve()): str(sample["split"]) for sample in samples
    }
    orientation_classes = {"horizontal": 0, "vertical": 1}
    color_classes = {"green": 0, "red": 1}

    _patch_tiny_dataset_loader(monkeypatch)

    factor_datasets: list[Path] = []
    for factor_name, class_map in (
        ("orientation", orientation_classes),
        ("color", color_classes),
    ):
        output_path = tmp_path / f"export_{factor_name}"
        worker = ExportWorker(
            image_paths=image_paths,
            labels=[class_map[str(sample[factor_name])] for sample in samples],
            output_path=output_path,
            format="ultralytics",
            class_names={value: key for key, value in class_map.items()},
            preset_splits_by_path=preset_splits,
        )

        payload = _run_export_worker(worker)
        assert payload["num_exported"] == len(samples)
        assert payload["source_split_by_path"] == preset_splits
        factor_datasets.append(output_path)

    for factor_index, dataset_dir in enumerate(factor_datasets):
        _set_global_seeds(4321 + factor_index)
        if trainer_kind == "tiny":
            spec = TrainingRunSpec(
                role=TrainingRole.CLASSIFY_MULTIHEAD_TINY,
                source_datasets=[],
                derived_dataset_dir=str(dataset_dir),
                base_model="",
                hyperparams=TrainingHyperParams(),
                device="cpu",
                seed=4321 + factor_index,
                augmentation_profile=AugmentationProfile(enabled=False),
                tiny_params=TinyHeadTailParams(
                    epochs=6,
                    batch=8,
                    lr=3e-3,
                    patience=6,
                    input_width=64,
                    input_height=64,
                    hidden_layers=1,
                    hidden_dim=48,
                    dropout=0.1,
                ),
            )
        else:
            spec = TrainingRunSpec(
                role=TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM,
                source_datasets=[],
                derived_dataset_dir=str(dataset_dir),
                base_model="",
                hyperparams=TrainingHyperParams(),
                device="cpu",
                seed=4321 + factor_index,
                augmentation_profile=AugmentationProfile(enabled=False),
                custom_params=CustomCNNParams(
                    backbone="tinyclassifier",
                    epochs=6,
                    batch=8,
                    lr=3e-3,
                    patience=6,
                    input_width=64,
                    input_height=64,
                    hidden_layers=1,
                    hidden_dim=48,
                    dropout=0.1,
                ),
            )
        run_dir = tmp_path / f"run_factor_{factor_index}"
        result = run_training(spec, run_dir)

        assert result["success"] is True
        assert Path(result["artifact_path"]).exists()
        test_accuracy = _evaluate_dataset_with_checkpoint(
            Path(result["artifact_path"]),
            dataset_dir,
        )
        assert test_accuracy >= threshold
