#!/usr/bin/env python3
"""Run reproducible ClassKit architecture sweeps against a project bundle.

Exports the labeled project data using the same split and label-expansion
semantics as the current ClassKit training path, trains one or more
lightweight classifier variants, and evaluates the saved artifact on the
exported held-out split.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2
import numpy as np

from hydra_suite.classkit.core.export.splits import (
    build_label_expansion_records,
    build_label_expansion_split_key,
    build_training_dataset_splits,
)
from hydra_suite.classkit.core.train.metrics import compute_metrics
from hydra_suite.classkit.gui.project import (
    classkit_config_path,
    classkit_db_path,
    prepare_project_directory,
)
from hydra_suite.classkit.jobs.task_workers import ExportWorker
from hydra_suite.classkit.store.db import ClassKitDB
from hydra_suite.core.identity.classification.backend import ClassifierBackend
from hydra_suite.training.contracts import (
    AugmentationProfile,
    CustomCNNParams,
    TinyHeadTailParams,
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)
from hydra_suite.training.runner import run_training

DEFAULT_TRIALS = [
    "tiny_s",
    "tiny_m",
    "tiny_l",
    "tiny_l_balanced",
    "mobilenet_v3_small",
    "shufflenet_v2_x1_0",
    "efficientnet_b0",
    "resnet18",
]


TRIAL_LIBRARY: dict[str, dict[str, Any]] = {
    "tiny_s": {
        "custom_backbone": "tinyclassifier",
        "tiny_preset": "small",
        "batch": 128,
    },
    "tiny_m": {
        "custom_backbone": "tinyclassifier",
        "tiny_preset": "medium",
        "batch": 128,
    },
    "tiny_l": {
        "custom_backbone": "tinyclassifier",
        "tiny_preset": "large",
        "batch": 128,
    },
    "tiny_l_balanced": {
        "custom_backbone": "tinyclassifier",
        "tiny_preset": "large",
        "tiny_rebalance_mode": "weighted_loss",
        "tiny_rebalance_power": 0.5,
        "tiny_label_smoothing": 0.05,
        "batch": 128,
    },
    "tiny_l_noexp": {
        "custom_backbone": "tinyclassifier",
        "tiny_preset": "large",
        "label_expansion": {},
        "batch": 128,
    },
    "mobilenet_v3_small": {
        "custom_backbone": "mobilenet_v3_small",
        "custom_input_size": 224,
        "custom_fine_tune_method": "full_finetune",
        "custom_trainable_layers": -1,
        "custom_backbone_lr_scale": 0.1,
        "batch": 64,
    },
    "shufflenet_v2_x1_0": {
        "custom_backbone": "shufflenet_v2_x1_0",
        "custom_input_size": 224,
        "custom_fine_tune_method": "full_finetune",
        "custom_trainable_layers": -1,
        "custom_backbone_lr_scale": 0.1,
        "batch": 64,
    },
    "efficientnet_b0": {
        "custom_backbone": "efficientnet_b0",
        "custom_input_size": 224,
        "custom_fine_tune_method": "full_finetune",
        "custom_trainable_layers": -1,
        "custom_backbone_lr_scale": 0.1,
        "batch": 32,
    },
    "resnet18": {
        "custom_backbone": "resnet18",
        "custom_input_size": 224,
        "custom_fine_tune_method": "full_finetune",
        "custom_trainable_layers": -1,
        "custom_backbone_lr_scale": 0.1,
        "batch": 32,
    },
}


def _load_project_state(
    project_dir: Path,
) -> tuple[dict[str, Any], list[tuple[Path, str]]]:
    prepare_project_directory(project_dir)
    config = json.loads(classkit_config_path(project_dir).read_text(encoding="utf-8"))
    db = ClassKitDB(classkit_db_path(project_dir))
    image_paths = [Path(path) for path in db.get_all_image_paths()]
    labels = db.get_all_labels()
    labeled_pairs = []
    for path, label in zip(image_paths, labels):
        label_name = str(label).strip() if label is not None else ""
        if label_name:
            labeled_pairs.append((path, label_name))
    if not labeled_pairs:
        raise RuntimeError(f"No labeled samples found in {project_dir}")
    return config, labeled_pairs


def _effective_class_names(
    project_classes: list[str], labels_str: list[str]
) -> list[str]:
    ordered = [str(name) for name in project_classes]
    seen = set(ordered)
    for label in labels_str:
        if label not in seen:
            ordered.append(label)
            seen.add(label)
    return ordered


def _build_export_context(
    images: list[Path],
    labels_str: list[str],
    class_names: list[str],
    settings: dict[str, Any],
) -> dict[str, Any]:
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    int_labels = [class_to_idx[label] for label in labels_str]
    class_names_int = {idx: name for name, idx in class_to_idx.items()}
    label_expansion = settings.get("label_expansion") or {}
    planned_records = build_label_expansion_records(
        labels_str,
        label_expansion=label_expansion,
        groups=None,
        known_labels=class_names,
    )
    planned_labels = [str(record["label"]) for record in planned_records]
    planned_splits, _used_group_fallback = build_training_dataset_splits(
        planned_labels,
        strategy=str(settings.get("split_strategy", "stratified")),
        val_fraction=float(settings.get("val_fraction", 0.2)),
        test_fraction=float(settings.get("test_fraction", 0.0)),
        groups=None,
    )

    source_split_by_path: dict[str, str] = {}
    expanded_split_by_key: dict[str, str] = {}
    source_splits: list[str] = []
    for record, split in zip(planned_records, planned_splits):
        source_path = images[int(record["source_index"])]
        if bool(record.get("is_expanded")):
            expanded_split_by_key[
                build_label_expansion_split_key(
                    source_path,
                    str(record.get("axis") or ""),
                    str(record["label"]),
                )
            ] = str(split)
            continue
        source_split_by_path[str(source_path.resolve())] = str(split)
        source_splits.append(str(split))

    return {
        "images": images,
        "labels_str": labels_str,
        "int_labels": int_labels,
        "class_names_int": class_names_int,
        "source_split_by_path": source_split_by_path,
        "expanded_split_by_key": expanded_split_by_key,
        "source_splits": source_splits,
    }


def _export_dataset(
    dataset_dir: Path, context: dict[str, Any], settings: dict[str, Any]
) -> dict[str, Any]:
    worker = ExportWorker(
        image_paths=context["images"],
        labels=context["int_labels"],
        output_path=dataset_dir,
        format="ultralytics",
        class_names=context["class_names_int"],
        split_strategy=str(settings.get("split_strategy", "stratified")),
        val_fraction=float(settings.get("val_fraction", 0.2)),
        test_fraction=float(settings.get("test_fraction", 0.0)),
        force_monochrome=bool(settings.get("monochrome", False)),
        label_expansion=settings.get("label_expansion") or {},
        preset_splits_by_path=context["source_split_by_path"],
        preset_expanded_splits_by_key=context["expanded_split_by_key"],
    )
    try:
        worker._prepare_export_workspace()
        image_paths, labels, splits, class_names = worker._collect_valid_labels()
        image_paths, labels, splits = worker._apply_label_expansion(
            image_paths, labels, splits, class_names
        )
        image_paths, labels, splits = worker._apply_monochrome_mode(
            image_paths, labels, splits
        )
        worker._export_dataset(image_paths, labels, splits, class_names)
    finally:
        tmpdir = getattr(worker, "_expansion_tmpdir", None)
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    split_counts: dict[str, dict[str, int]] = {}
    for split_name, label in zip(splits, labels):
        split_counts.setdefault(split_name, Counter())
        split_counts[split_name][class_names[label]] += 1
    return {
        "num_exported": len(image_paths),
        "split_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_counts.items())
        },
    }


def _make_spec(settings: dict[str, Any], dataset_dir: Path) -> TrainingRunSpec:
    aug = AugmentationProfile(
        enabled=True,
        flipud=float(settings.get("flipud", 0.0)),
        fliplr=float(settings.get("fliplr", 0.0)),
        hue=float(settings.get("hue", 0.0)),
        saturation=float(settings.get("saturation", 0.0)),
        brightness=float(settings.get("brightness", 0.0)),
        contrast=float(settings.get("contrast", 0.0)),
        monochrome=bool(settings.get("monochrome", False)),
        args={
            key: value
            for key, value in {
                "flipud": settings.get("flipud", 0.0),
                "fliplr": settings.get("fliplr", 0.0),
                "hsv_h": settings.get("hue", 0.0),
                "hsv_s": settings.get("saturation", 0.0),
                "hsv_v": settings.get("brightness", 0.0),
            }.items()
            if float(value) > 0.0
        },
        label_expansion=settings.get("label_expansion") or {},
    )

    return TrainingRunSpec(
        role=TrainingRole.CLASSIFY_FLAT_CUSTOM,
        source_datasets=[],
        derived_dataset_dir=str(dataset_dir),
        base_model="",
        hyperparams=TrainingHyperParams(
            epochs=int(settings.get("epochs", 50)),
            batch=int(settings.get("batch", 32)),
            lr0=float(settings.get("lr", 1e-3)),
            patience=int(settings.get("patience", 10)),
        ),
        tiny_params=TinyHeadTailParams(
            epochs=int(settings.get("epochs", 50)),
            batch=int(settings.get("batch", 32)),
            lr=float(settings.get("lr", 1e-3)),
            patience=int(settings.get("patience", 10)),
            tiny_preset=str(settings.get("tiny_preset", "medium")),
            hidden_layers=int(settings.get("tiny_layers", 1)),
            hidden_dim=int(settings.get("tiny_dim", 96)),
            dropout=float(settings.get("tiny_dropout", 0.1)),
            input_width=int(settings.get("tiny_width", 128)),
            input_height=int(settings.get("tiny_height", 64)),
            class_rebalance_mode=str(settings.get("tiny_rebalance_mode", "none")),
            class_rebalance_power=float(settings.get("tiny_rebalance_power", 1.0)),
            label_smoothing=float(settings.get("tiny_label_smoothing", 0.0)),
        ),
        custom_params=CustomCNNParams(
            backbone=str(settings.get("custom_backbone", "tinyclassifier")),
            fine_tune_method=str(settings.get("custom_fine_tune_method", "head_only")),
            trainable_layers=int(settings.get("custom_trainable_layers", 0)),
            backbone_lr_scale=float(settings.get("custom_backbone_lr_scale", 0.1)),
            layerwise_lr_decay=float(settings.get("custom_layerwise_lr_decay", 0.75)),
            gradual_unfreeze_interval=int(
                settings.get("custom_gradual_unfreeze_interval", 5)
            ),
            input_size=int(settings.get("custom_input_size", 224)),
            epochs=int(settings.get("epochs", 50)),
            batch=int(settings.get("batch", 32)),
            lr=float(settings.get("lr", 1e-3)),
            patience=int(settings.get("patience", 10)),
            weight_decay=1e-2,
            tiny_preset=str(settings.get("tiny_preset", "medium")),
            hidden_layers=int(settings.get("tiny_layers", 1)),
            hidden_dim=int(settings.get("tiny_dim", 96)),
            dropout=float(settings.get("tiny_dropout", 0.1)),
            input_width=int(settings.get("tiny_width", 128)),
            input_height=int(settings.get("tiny_height", 64)),
            label_smoothing=float(settings.get("tiny_label_smoothing", 0.0)),
            class_rebalance_mode=str(settings.get("tiny_rebalance_mode", "none")),
            class_rebalance_power=float(settings.get("tiny_rebalance_power", 1.0)),
        ),
        device=str(settings.get("device", "cpu")),
        training_space="original",
        augmentation_profile=aug,
    )


def _iter_split_records(dataset_dir: Path, split_name: str) -> list[tuple[Path, str]]:
    split_dir = dataset_dir / split_name
    if not split_dir.exists():
        return []
    records = []
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tif",
                ".tiff",
            }:
                records.append((image_path, class_dir.name))
    return records


def _evaluate_artifact(
    artifact_path: Path, dataset_dir: Path, compute_runtime: str, batch_size: int = 64
) -> dict[str, Any]:
    split_name = "test" if (dataset_dir / "test").exists() else "val"
    records = _iter_split_records(dataset_dir, split_name)
    if not records:
        return {
            "split": split_name,
            "num_samples": 0,
            "error": f"No {split_name} samples exported",
        }

    backend = ClassifierBackend(str(artifact_path), compute_runtime=compute_runtime)
    try:
        class_names = list(backend.metadata.class_names_per_factor[0])
        label_to_idx = {name: idx for idx, name in enumerate(class_names)}
        y_true: list[int] = []
        y_pred: list[int] = []
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            crops = []
            batch_true = []
            for image_path, class_name in batch_records:
                img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                idx = label_to_idx.get(class_name)
                if idx is None:
                    continue
                crops.append(img)
                batch_true.append(idx)
            if not crops:
                continue
            probs = backend.predict_batch(crops)
            for truth, per_factor in zip(batch_true, probs):
                y_true.append(int(truth))
                y_pred.append(int(np.argmax(per_factor[0])))

        if not y_true:
            return {
                "split": split_name,
                "num_samples": 0,
                "error": "No evaluable held-out samples",
            }
        metrics = compute_metrics(
            predictions=np.asarray(y_pred, dtype=np.int64),
            labels=np.asarray(y_true, dtype=np.int64),
            class_names=class_names,
        )
        return {
            "split": split_name,
            "num_samples": metrics.num_samples,
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "weighted_f1": metrics.weighted_f1,
            "per_class": [
                {
                    "class_name": item.class_name,
                    "precision": item.precision,
                    "recall": item.recall,
                    "f1": item.f1,
                    "support": item.support,
                }
                for item in metrics.per_class
            ],
            "confusion_matrix": metrics.confusion_matrix.tolist(),
        }
    finally:
        backend.close()


def _trial_settings(
    base_settings: dict[str, Any], trial_name: str, args: argparse.Namespace
) -> dict[str, Any]:
    if trial_name not in TRIAL_LIBRARY:
        raise KeyError(
            f"Unknown trial {trial_name!r}. Known trials: {sorted(TRIAL_LIBRARY)}"
        )
    settings = dict(base_settings)
    settings.update(TRIAL_LIBRARY[trial_name])
    settings["mode"] = "flat_custom"
    settings["device"] = args.device
    settings["epochs"] = (
        args.epochs if args.epochs is not None else settings.get("epochs", 50)
    )
    settings["batch"] = (
        args.batch if args.batch is not None else settings.get("batch", 32)
    )
    settings["lr"] = args.lr if args.lr is not None else settings.get("lr", 1e-3)
    settings["patience"] = (
        args.patience if args.patience is not None else settings.get("patience", 10)
    )
    settings["split_strategy"] = args.split_strategy or settings.get(
        "split_strategy", "random"
    )
    settings["val_fraction"] = (
        args.val_fraction
        if args.val_fraction is not None
        else settings.get("val_fraction", 0.2)
    )
    settings["test_fraction"] = (
        args.test_fraction
        if args.test_fraction is not None
        else settings.get("test_fraction", 0.0)
    )
    if args.disable_label_expansion:
        settings["label_expansion"] = {}
    return settings


def _summarize_trial(result: dict[str, Any]) -> str:
    train_metric = result.get("best_val_acc")
    eval_metric = (result.get("evaluation") or {}).get("accuracy")
    macro_f1 = (result.get("evaluation") or {}).get("macro_f1")
    status = "ok" if result.get("success") else "failed"
    return (
        f"[{status}] {result['trial']} backbone={result['backbone']} "
        f"best_val_acc={train_metric!r} heldout_acc={eval_metric!r} macro_f1={macro_f1!r}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        type=Path,
        required=True,
        help="Path to the ClassKit project bundle",
    )
    parser.add_argument(
        "--trial",
        dest="trials",
        action="append",
        default=[],
        help=f"Trial name to run. Can be repeated. Defaults to {', '.join(DEFAULT_TRIALS)}",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--split-strategy", default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--test-fraction", type=float, default=None)
    parser.add_argument("--disable-label-expansion", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Directory where per-trial exports and checkpoints are written",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_dir = args.project.expanduser().resolve()
    config, labeled_pairs = _load_project_state(project_dir)
    images = [path.resolve() for path, _label in labeled_pairs]
    labels_str = [label for _path, label in labeled_pairs]
    project_classes = [str(name) for name in config.get("classes") or []]
    effective_classes = _effective_class_names(project_classes, labels_str)
    base_settings = dict(config.get("last_training_settings") or {})
    trials = args.trials or list(DEFAULT_TRIALS)
    run_root = (
        args.run_root.expanduser().resolve()
        if args.run_root is not None
        else project_dir / ".classkit_runs" / f"sweep_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "project": str(project_dir),
        "project_classes": project_classes,
        "effective_classes": effective_classes,
        "labeled_total": len(labeled_pairs),
        "label_counts": dict(sorted(Counter(labels_str).items())),
        "trials": [],
    }

    for trial_name in trials:
        settings = _trial_settings(base_settings, trial_name, args)
        context = _build_export_context(images, labels_str, effective_classes, settings)
        trial_dir = run_root / trial_name
        dataset_dir = trial_dir / "export"
        train_dir = trial_dir / "train"
        logs: list[str] = []
        result: dict[str, Any] = {
            "trial": trial_name,
            "backbone": str(settings.get("custom_backbone", "tinyclassifier")),
            "settings": {
                "epochs": settings.get("epochs"),
                "batch": settings.get("batch"),
                "lr": settings.get("lr"),
                "device": settings.get("device"),
                "split_strategy": settings.get("split_strategy"),
                "val_fraction": settings.get("val_fraction"),
                "test_fraction": settings.get("test_fraction"),
                "label_expansion": settings.get("label_expansion") or {},
                "tiny_preset": settings.get("tiny_preset"),
                "tiny_rebalance_mode": settings.get("tiny_rebalance_mode"),
                "tiny_rebalance_power": settings.get("tiny_rebalance_power"),
                "tiny_label_smoothing": settings.get("tiny_label_smoothing"),
            },
            "success": False,
        }
        try:
            export_info = _export_dataset(dataset_dir, context, settings)
            result["export"] = export_info
            spec = _make_spec(settings, dataset_dir)
            train_result = run_training(
                spec,
                train_dir,
                log_cb=lambda msg, _logs=logs: _logs.append(str(msg)),
            )
            result.update(
                {
                    "success": bool(train_result.get("success")),
                    "artifact_path": train_result.get("artifact_path", ""),
                    "metrics_path": train_result.get("metrics_path", ""),
                    "best_val_acc": train_result.get("best_val_acc"),
                }
            )
            if train_result.get("artifact_path"):
                result["evaluation"] = _evaluate_artifact(
                    Path(str(train_result["artifact_path"])),
                    dataset_dir,
                    compute_runtime=str(settings.get("device", "cpu")),
                )
            else:
                result["evaluation"] = {"error": "No artifact produced"}
        except Exception as exc:
            result["error"] = str(exc)
        result["log_tail"] = logs[-20:]
        summary["trials"].append(result)
        print(_summarize_trial(result))

    ranked = sorted(
        summary["trials"],
        key=lambda item: (
            float((item.get("evaluation") or {}).get("accuracy", -1.0)),
            float(item.get("best_val_acc") or -1.0),
        ),
        reverse=True,
    )
    summary["ranked_trials"] = [item["trial"] for item in ranked]

    output_json = args.output_json
    if output_json is None:
        output_json = run_root / "summary.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
