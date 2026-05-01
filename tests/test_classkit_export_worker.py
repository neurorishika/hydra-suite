from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")

from hydra_suite.classkit.core.export.splits import (
    build_dataset_splits,
    build_label_expansion_split_key,
    build_training_dataset_splits,
)
from hydra_suite.classkit.jobs.task_workers import ExportWorker


def _run_worker_and_collect_error(worker: ExportWorker) -> list[str]:
    errors: list[str] = []
    worker.signals.error.connect(errors.append)
    worker.run()
    return errors


def test_label_expansion_auto_creates_temp_dir(tmp_path: Path) -> None:
    """Label expansion should auto-create a temp dir when none is provided."""
    worker = ExportWorker(
        image_paths=[tmp_path / "img_0.jpg"],
        labels=[0],
        output_path=tmp_path / "out.csv",
        format="csv",
        class_names={0: "left"},
        label_expansion={"fliplr": {"left": "right"}},
    )

    errors = _run_worker_and_collect_error(worker)

    assert worker.temp_dir is not None
    assert errors == []


def test_export_worker_no_labeled_samples(
    tmp_path: Path,
) -> None:
    worker = ExportWorker(
        image_paths=[tmp_path / "img_0.jpg"],
        labels=[-1],  # no valid labeled samples
        output_path=tmp_path / "out.csv",
        format="csv",
        class_names={0: "left"},
        label_expansion={"fliplr": {"left": "right"}},
    )

    errors = _run_worker_and_collect_error(worker)

    assert errors
    assert "No labeled samples found to export" in errors[0]


def test_export_worker_collect_valid_labels_uses_stratified_split(
    tmp_path: Path,
) -> None:
    worker = ExportWorker(
        image_paths=[tmp_path / f"img_{idx}.jpg" for idx in range(6)],
        labels=[0, 0, 0, 0, 1, 1],
        output_path=tmp_path / "out.csv",
        format="csv",
        class_names={0: "left", 1: "right"},
        val_fraction=0.33,
    )

    _image_paths, labels, splits, _class_names = worker._collect_valid_labels()

    label_split_counts = Counter(zip(labels, splits))
    assert len(labels) == 6
    assert splits.count("val") == 2
    assert label_split_counts[(0, "val")] == 1
    assert label_split_counts[(1, "val")] == 1


def test_export_worker_split_planning_ignores_unlabeled_items(tmp_path: Path) -> None:
    worker = ExportWorker(
        image_paths=[tmp_path / f"img_{idx}.jpg" for idx in range(8)],
        labels=[0, 0, 0, 0, 1, 1, -1, -1],
        output_path=tmp_path / "out.csv",
        format="csv",
        class_names={0: "left", 1: "right"},
        val_fraction=0.33,
    )

    image_paths, labels, splits, _class_names = worker._collect_valid_labels()

    assert len(image_paths) == 6
    assert len(labels) == 6
    assert len(splits) == 6
    assert splits.count("val") == 2


def test_export_worker_split_planning_uses_requested_strategy(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def _fake_build_dataset_splits(
        labels, *, strategy, val_fraction, test_fraction, seed=42
    ):
        captured["labels"] = list(labels)
        captured["strategy"] = strategy
        captured["val_fraction"] = val_fraction
        captured["test_fraction"] = test_fraction
        return ["train"] * len(labels)

    monkeypatch.setattr(
        "hydra_suite.classkit.jobs.task_workers.build_dataset_splits",
        _fake_build_dataset_splits,
    )

    worker = ExportWorker(
        image_paths=[tmp_path / f"img_{idx}.jpg" for idx in range(4)],
        labels=[0, 0, 1, 1],
        output_path=tmp_path / "out.csv",
        format="csv",
        class_names={0: "left", 1: "right"},
        split_strategy="random",
        val_fraction=0.25,
    )

    _image_paths, labels, splits, _class_names = worker._collect_valid_labels()

    assert labels == [0, 0, 1, 1]
    assert splits == ["train", "train", "train", "train"]
    assert captured == {
        "labels": [0, 0, 1, 1],
        "strategy": "random",
        "val_fraction": 0.25,
        "test_fraction": 0.0,
    }


def test_grouped_dataset_splits_keep_related_samples_together() -> None:
    labels = ["left", "left", "left", "left", "right", "right", "right", "right"]
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]

    splits = build_dataset_splits(
        labels,
        strategy="stratified",
        val_fraction=0.25,
        test_fraction=0.25,
        groups=groups,
    )

    assert splits.count("train") == 4
    assert splits.count("val") == 2
    assert splits.count("test") == 2
    for group in sorted(set(groups)):
        assigned = {split for split, key in zip(splits, groups) if key == group}
        assert len(assigned) == 1


def test_grouped_random_splits_do_not_consume_only_group() -> None:
    labels = ["left"] * 10
    groups = ["session-a"] * 10

    splits = build_dataset_splits(
        labels,
        strategy="random",
        val_fraction=0.2,
        test_fraction=0.0,
        groups=groups,
    )

    assert splits == ["train"] * 10


def test_grouped_stratified_splits_do_not_consume_only_group() -> None:
    labels = ["left"] * 10
    groups = ["session-a"] * 10

    splits = build_dataset_splits(
        labels,
        strategy="stratified",
        val_fraction=0.2,
        test_fraction=0.0,
        groups=groups,
    )

    assert splits == ["train"] * 10


def test_training_split_builder_falls_back_when_grouping_blocks_holdout() -> None:
    labels = ["left"] * 10
    groups = ["session-a"] * 10

    splits, used_group_fallback = build_training_dataset_splits(
        labels,
        strategy="random",
        val_fraction=0.2,
        test_fraction=0.0,
        groups=groups,
    )

    assert used_group_fallback is True
    assert splits.count("train") == 8
    assert splits.count("val") == 2


def test_export_worker_uses_preset_splits_by_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_paths = [tmp_path / f"img_{idx}.jpg" for idx in range(4)]
    preset = {
        str(path.resolve()): split
        for path, split in zip(image_paths, ["train", "test", "val", "train"])
    }

    def _unexpected_split_build(*args, **kwargs):
        raise AssertionError("build_dataset_splits should not be called")

    monkeypatch.setattr(
        "hydra_suite.classkit.jobs.task_workers.build_dataset_splits",
        _unexpected_split_build,
    )

    worker = ExportWorker(
        image_paths=image_paths,
        labels=[0, 0, 1, 1],
        output_path=tmp_path / "out.csv",
        format="csv",
        class_names={0: "left", 1: "right"},
        preset_splits_by_path=preset,
    )

    collected_paths, labels, splits, _class_names = worker._collect_valid_labels()

    assert labels == [0, 0, 1, 1]
    assert splits == ["train", "test", "val", "train"]
    assert [str(path.resolve()) for path in collected_paths] == list(preset.keys())


def test_multihead_export_worker_filters_unknown_factor_labels(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hydra_suite.classkit.gui.main_window import MainWindow

    recorded_runs: list[dict[str, object]] = []

    class _Emitter:
        def emit(self, *args, **kwargs) -> None:
            return None

    class _FakeExportWorker:
        def __init__(self, *args, **kwargs) -> None:
            self.image_paths = list(kwargs.get("image_paths") or [])
            self.labels = list(kwargs.get("labels") or [])
            self.class_names = dict(kwargs.get("class_names") or {})
            self.output_path = kwargs.get("output_path")
            self.signals = SimpleNamespace(
                started=_Emitter(),
                progress=_Emitter(),
                success=_Emitter(),
                error=_Emitter(),
                finished=_Emitter(),
            )

        def run(self) -> None:
            recorded_runs.append(
                {
                    "image_paths": list(self.image_paths),
                    "labels": list(self.labels),
                    "class_names": dict(self.class_names),
                    "output_path": self.output_path,
                }
            )

    monkeypatch.setattr(
        "hydra_suite.classkit.jobs.task_workers.ExportWorker",
        _FakeExportWorker,
    )

    window = MainWindow()
    image_paths = [tmp_path / f"img_{idx}.png" for idx in range(3)]
    window.image_paths = image_paths

    scheme = SimpleNamespace(
        factors=[SimpleNamespace(name="color"), SimpleNamespace(name="side")],
        decode_label=lambda label: tuple(str(label).split("_")),
    )
    context = {
        "settings": {},
        "images": image_paths,
        "run_dir": tmp_path / "export_root",
        "source_split_by_path": {},
        "expanded_split_by_key": None,
        "labels_str": ["red_left", "unknown_right", "blue_unknown"],
    }

    worker = window._create_multihead_export_worker(context, scheme)
    worker.run()

    assert len(recorded_runs) == 2
    assert [len(run["image_paths"]) for run in recorded_runs] == [2, 2]
    assert set(recorded_runs[0]["class_names"].values()) == {"blue", "red"}
    assert set(recorded_runs[1]["class_names"].values()) == {"left", "right"}


def test_export_worker_force_monochrome_materializes_grayscale_copies(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "color.png"
    Image.new("RGB", (12, 10), color=(200, 80, 20)).save(image_path)

    worker = ExportWorker(
        image_paths=[image_path],
        labels=[0],
        output_path=tmp_path / "out.csv",
        format="csv",
        class_names={0: "left"},
        force_monochrome=True,
    )

    worker._prepare_export_workspace()
    image_paths, labels, splits, _class_names = worker._collect_valid_labels()
    converted_paths, converted_labels, converted_splits = worker._apply_monochrome_mode(
        image_paths, labels, splits
    )

    converted = np.asarray(Image.open(converted_paths[0]).convert("RGB"))

    assert converted_paths[0] != image_path
    assert converted_labels == labels
    assert converted_splits == splits
    assert np.array_equal(converted[..., 0], converted[..., 1])
    assert np.array_equal(converted[..., 1], converted[..., 2])


def test_export_worker_label_expansion_respects_preset_expanded_split_keys(
    tmp_path: Path,
) -> None:
    pytest.importorskip("cv2")

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (12, 10), color=(200, 80, 20)).save(image_path)

    worker = ExportWorker(
        image_paths=[image_path],
        labels=[0],
        output_path=tmp_path / "out.csv",
        format="csv",
        class_names={0: "left", 1: "right"},
        label_expansion={"fliplr": {"left": "right"}},
        preset_splits_by_path={str(image_path.resolve()): "train"},
        preset_expanded_splits_by_key={
            build_label_expansion_split_key(image_path, "fliplr", "right"): "val"
        },
    )

    worker._prepare_export_workspace()
    image_paths, labels, splits, class_names = worker._collect_valid_labels()
    expanded_paths, expanded_labels, expanded_splits = worker._apply_label_expansion(
        image_paths,
        labels,
        splits,
        class_names,
    )

    assert expanded_labels == [0, 1]
    assert expanded_splits == ["train", "val"]
    assert len(expanded_paths) == 2


def test_export_worker_ultralytics_allows_empty_validation_split(
    tmp_path: Path,
) -> None:
    image_paths = []
    for idx, color in enumerate(((200, 80, 20), (20, 120, 220))):
        image_path = tmp_path / f"sample_{idx}.png"
        Image.new("RGB", (12, 10), color=color).save(image_path)
        image_paths.append(image_path)

    output_path = tmp_path / "ultralytics_out"
    worker = ExportWorker(
        image_paths=image_paths,
        labels=[0, 0],
        output_path=output_path,
        format="ultralytics",
        class_names={0: "left"},
        val_fraction=0.0,
    )

    errors = _run_worker_and_collect_error(worker)

    train_files = sorted((output_path / "train" / "left").glob("*.png"))
    val_files = list((output_path / "val").rglob("*.png"))

    assert errors == []
    assert len(train_files) == 2
    assert val_files == []


def test_export_worker_imagefolder_preserves_duplicate_basenames(
    tmp_path: Path,
) -> None:
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    source_a.mkdir()
    source_b.mkdir()

    image_a = source_a / "sample.png"
    image_b = source_b / "sample.png"
    Image.new("RGB", (12, 10), color=(200, 80, 20)).save(image_a)
    Image.new("RGB", (12, 10), color=(20, 120, 220)).save(image_b)

    output_path = tmp_path / "imagefolder_out"
    worker = ExportWorker(
        image_paths=[image_a, image_b],
        labels=[0, 0],
        output_path=output_path,
        format="imagefolder",
        class_names={0: "left"},
        val_fraction=0.0,
    )

    errors = _run_worker_and_collect_error(worker)

    train_files = sorted((output_path / "train" / "left").glob("*.png"))

    assert errors == []
    assert len(train_files) == 2
    assert {path.name for path in train_files} == {"sample.png", "sample_1.png"}
