from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")

task_workers = pytest.importorskip("hydra_suite.classkit.jobs.task_workers")
TinyCNNInferenceWorker = task_workers.TinyCNNInferenceWorker
TorchvisionInferenceWorker = task_workers.TorchvisionInferenceWorker
YoloInferenceWorker = task_workers.YoloInferenceWorker


def test_tiny_inference_loader_forces_monochrome(tmp_path: Path) -> None:
    image_path = tmp_path / "color.png"
    Image.new("RGB", (14, 10), color=(200, 80, 20)).save(image_path)

    batch = TinyCNNInferenceWorker._load_batch_images(
        [image_path],
        8,
        8,
        force_monochrome=True,
    )

    assert batch.shape == (1, 3, 8, 8)
    assert np.allclose(batch[0, 0], batch[0, 1])
    assert np.allclose(batch[0, 1], batch[0, 2])


def test_torchvision_inference_loader_forces_monochrome(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    image_path = tmp_path / "color.png"
    Image.new("RGB", (14, 10), color=(200, 80, 20)).save(image_path)

    worker = TorchvisionInferenceWorker(
        model_path=Path("/tmp/fake_model.pth"),
        image_paths=[image_path],
        class_names=["left"],
        input_size=8,
        force_monochrome=True,
    )

    def _tensorize(img):
        array = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(array)

    batch = worker._load_batch_tensors([image_path], _tensorize)

    assert batch.shape == (1, 3, 10, 14)
    assert np.allclose(batch[0, 0], batch[0, 1])
    assert np.allclose(batch[0, 1], batch[0, 2])


def test_torchvision_monochrome_transform_preserves_equal_channels(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")

    image_path = tmp_path / "color.png"
    Image.new("RGB", (14, 10), color=(200, 80, 20)).save(image_path)

    worker = TorchvisionInferenceWorker(
        model_path=Path("/tmp/fake_model.pth"),
        image_paths=[image_path],
        class_names=["left"],
        input_size=8,
        force_monochrome=True,
    )

    transformed = worker._build_transform()(Image.open(image_path).convert("RGB"))

    assert transformed.shape == (3, 8, 8)
    assert torch.allclose(transformed[0], transformed[1])
    assert torch.allclose(transformed[1], transformed[2])


def test_yolo_inference_loader_forces_monochrome(tmp_path: Path) -> None:
    image_path = tmp_path / "color.png"
    Image.new("RGB", (14, 10), color=(200, 80, 20)).save(image_path)

    batch = YoloInferenceWorker._prepare_batch_input(
        [image_path],
        force_monochrome=True,
    )

    assert len(batch) == 1
    assert batch[0].shape == (10, 14, 3)
    assert np.array_equal(batch[0][..., 0], batch[0][..., 1])
    assert np.array_equal(batch[0][..., 1], batch[0][..., 2])
