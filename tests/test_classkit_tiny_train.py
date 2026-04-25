# tests/test_classkit_tiny_train.py
from pathlib import Path

import pytest


def _make_dummy_dataset(root: Path, classes: list[str], n_per_class: int = 4):
    """Create minimal image-folder dataset with tiny solid-color images."""
    import numpy as np

    cv2 = pytest.importorskip("cv2")

    for split in ("train", "val"):
        for cls in classes:
            d = root / split / cls
            d.mkdir(parents=True)
            for i in range(n_per_class):
                img = (np.random.rand(32, 32, 3) * 255).astype("uint8")
                cv2.imwrite(str(d / f"img_{i}.png"), img)


def test_iter_classify_samples_general(tmp_path):
    from hydra_suite.training.runner import _iter_classify_samples

    _make_dummy_dataset(tmp_path, ["cat", "dog", "bird"])
    samples = list(_iter_classify_samples(tmp_path, "train"))
    assert len(samples) == 12  # 3 classes × 4 images
    labels = {s[1] for s in samples}
    assert labels == {0, 1, 2}


def test_iter_classify_samples_returns_sorted_class_order(tmp_path):
    from hydra_suite.training.runner import _iter_classify_samples

    _make_dummy_dataset(tmp_path, ["zebra", "apple", "mango"])
    samples = list(_iter_classify_samples(tmp_path, "train"))
    # Classes sorted alphabetically: apple=0, mango=1, zebra=2
    class_dir_names = sorted(["apple", "mango", "zebra"])
    label_map = {name: i for i, name in enumerate(class_dir_names)}
    for img_path, label in samples:
        expected = label_map[img_path.parent.name]
        assert label == expected


def test_iter_classify_samples_empty_split(tmp_path):
    from hydra_suite.training.runner import _iter_classify_samples

    # Non-existent split should yield nothing, not raise
    samples = list(_iter_classify_samples(tmp_path, "train"))
    assert samples == []


def test_build_class_to_idx_is_stable_across_splits(tmp_path):
    from hydra_suite.training.runner import _build_class_to_idx

    # Train has three classes, val has only one. Mapping must remain global.
    for cls in ("ant", "bee", "cat"):
        (tmp_path / "train" / cls).mkdir(parents=True)
    (tmp_path / "val" / "cat").mkdir(parents=True)

    class_to_idx = _build_class_to_idx(tmp_path)
    assert class_to_idx == {"ant": 0, "bee": 1, "cat": 2}


def test_iter_classify_samples_uses_shared_mapping_for_sparse_val_split(tmp_path):
    import numpy as np

    cv2 = pytest.importorskip("cv2")
    from hydra_suite.training.runner import _build_class_to_idx, _iter_classify_samples

    # Train covers all classes, val includes only one class.
    for cls in ("ant", "bee", "cat"):
        (tmp_path / "train" / cls).mkdir(parents=True)
        img = (np.random.rand(16, 16, 3) * 255).astype("uint8")
        cv2.imwrite(str(tmp_path / "train" / cls / "train.png"), img)

    (tmp_path / "val" / "cat").mkdir(parents=True)
    img = (np.random.rand(16, 16, 3) * 255).astype("uint8")
    cv2.imwrite(str(tmp_path / "val" / "cat" / "val.png"), img)

    class_to_idx = _build_class_to_idx(tmp_path)
    val_samples = list(_iter_classify_samples(tmp_path, "val", class_to_idx))

    assert len(val_samples) == 1
    # Old behavior incorrectly produced 0 here due to per-split enumeration.
    assert val_samples[0][1] == class_to_idx["cat"] == 2


def test_train_tiny_classify_runs(tmp_path):
    """Smoke test: tiny N-class CNN trains without error on minimal data."""
    pytest.importorskip("torch")
    pytest.importorskip("cv2")

    from hydra_suite.training.contracts import (
        TinyHeadTailParams,
        TrainingHyperParams,
        TrainingRole,
        TrainingRunSpec,
    )
    from hydra_suite.training.runner import run_training

    _make_dummy_dataset(tmp_path / "dataset", ["a", "b", "c"], n_per_class=4)

    spec = TrainingRunSpec(
        role=TrainingRole.CLASSIFY_FLAT_TINY,
        source_datasets=[],
        derived_dataset_dir=str(tmp_path / "dataset"),
        base_model="",
        hyperparams=TrainingHyperParams(),
        tiny_params=TinyHeadTailParams(epochs=2, batch=4),
    )

    result = run_training(spec, tmp_path / "run")
    assert result["success"] is True
    assert Path(result["artifact_path"]).exists()
    assert result["task"] == "tiny_classify"
    assert isinstance(result.get("best_val_acc"), float)


def test_train_tiny_classify_multihead_role(tmp_path):
    """CLASSIFY_MULTIHEAD_TINY uses same in-process trainer."""
    pytest.importorskip("torch")
    pytest.importorskip("cv2")

    from hydra_suite.training.contracts import (
        TinyHeadTailParams,
        TrainingHyperParams,
        TrainingRole,
        TrainingRunSpec,
    )
    from hydra_suite.training.runner import run_training

    _make_dummy_dataset(tmp_path / "dataset", ["x", "y"], n_per_class=4)

    spec = TrainingRunSpec(
        role=TrainingRole.CLASSIFY_MULTIHEAD_TINY,
        source_datasets=[],
        derived_dataset_dir=str(tmp_path / "dataset"),
        base_model="",
        hyperparams=TrainingHyperParams(),
        tiny_params=TinyHeadTailParams(epochs=2, batch=4),
    )

    result = run_training(spec, tmp_path / "run")
    assert result["success"] is True

    result = run_training(spec, tmp_path / "run")
    assert result["success"] is True


def test_extract_best_val_acc_from_results_csv(tmp_path):
    from hydra_suite.training.runner import _extract_best_val_acc_from_results_csv

    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "epoch,metrics/accuracy_top1,metrics/accuracy_top5\n"
        "1,0.60,0.88\n"
        "2,0.74,0.93\n"
        "3,0.71,0.94\n",
        encoding="utf-8",
    )

    best = _extract_best_val_acc_from_results_csv(csv_path)

    assert best == pytest.approx(0.74)


def test_train_tiny_classify_rebalance_modes_smoke(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("cv2")

    from hydra_suite.training.contracts import (
        TinyHeadTailParams,
        TrainingHyperParams,
        TrainingRole,
        TrainingRunSpec,
    )
    from hydra_suite.training.runner import run_training

    _make_dummy_dataset(tmp_path / "dataset", ["maj", "min"], n_per_class=3)

    spec = TrainingRunSpec(
        role=TrainingRole.CLASSIFY_FLAT_TINY,
        source_datasets=[],
        derived_dataset_dir=str(tmp_path / "dataset"),
        base_model="",
        hyperparams=TrainingHyperParams(),
        tiny_params=TinyHeadTailParams(
            epochs=2,
            batch=2,
            class_rebalance_mode="both",
            class_rebalance_power=1.0,
            label_smoothing=0.05,
        ),
    )

    result = run_training(spec, tmp_path / "run_rebalance")
    assert result["success"] is True
    assert isinstance(result.get("best_val_acc"), float)


def test_resolve_ignore_label_index_finds_unknown_class() -> None:
    from hydra_suite.training.runner import _resolve_ignore_label_index

    assert (
        _resolve_ignore_label_index(
            {"ant": 0, "bee": 1, "unknown": 2}, "unknown"
        )
        == 2
    )
    assert _resolve_ignore_label_index({"ant": 0, "bee": 1}, "unknown") is None


def test_compute_class_weights_skips_ignored_classes() -> None:
    from hydra_suite.training.runner import _compute_class_weights

    weights = _compute_class_weights(
        [0, 0, 1, 2],
        num_classes=3,
        mode="loss",
        power=1.0,
        ignored_class_indices={2},
    )

    assert weights is not None
    assert len(weights) == 3
    assert weights[0] > 0.0
    assert weights[1] > 0.0
    assert weights[2] == pytest.approx(0.0)
