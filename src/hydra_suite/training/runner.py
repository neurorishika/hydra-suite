"""Training execution utilities for MAT multi-role training."""

from __future__ import annotations

import csv
import dataclasses
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from .contracts import CustomCNNParams, TrainingRole, TrainingRunSpec

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]
CancelCheck = Callable[[], bool]


def _safe_log(cb: LogCallback | None, message: str) -> None:
    if cb is not None:
        cb(str(message))


def _looks_like_accuracy_column(name: str) -> bool:
    low = str(name or "").strip().lower()
    if not low:
        return False
    return ("acc" in low) or ("accuracy" in low)


def _accuracy_column_priority(name: str) -> tuple[int, str]:
    """Lower priority value means a better candidate for val/top-1 accuracy."""
    low = str(name or "").strip().lower()
    if not _looks_like_accuracy_column(low):
        return (99, low)
    if "top1" in low and "metrics" in low:
        return (0, low)
    if "top1" in low:
        return (1, low)
    if "val" in low:
        return (2, low)
    if "metrics" in low:
        return (3, low)
    return (4, low)


def _best_value_for_column(rows: list[dict], col: str) -> float | None:
    """Extract the maximum numeric value for a column across all rows."""
    values: list[float] = []
    for row in rows:
        raw = row.get(col)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            values.append(float(raw))
        except Exception:
            continue
    return max(values) if values else None


def _extract_best_val_acc_from_results_csv(metrics_csv_path: Path) -> float | None:
    """Extract best validation-like accuracy from Ultralytics results.csv."""
    if not metrics_csv_path.exists():
        return None
    try:
        with open(metrics_csv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            field_names = list(reader.fieldnames or [])
            candidate_cols = sorted(
                [c for c in field_names if _looks_like_accuracy_column(c)],
                key=_accuracy_column_priority,
            )
            if not candidate_cols:
                return None
            rows = list(reader)
            for col in candidate_cols:
                result = _best_value_for_column(rows, col)
                if result is not None:
                    return result
            return None
    except Exception:
        return None


def _ultralytics_task_for_role(role: TrainingRole) -> str:
    if role in (TrainingRole.OBB_DIRECT, TrainingRole.SEQ_CROP_OBB):
        return "obb"
    if role == TrainingRole.SEQ_DETECT:
        return "detect"
    if role in (
        TrainingRole.CLASSIFY_FLAT_YOLO,
        TrainingRole.CLASSIFY_MULTIHEAD_YOLO,
    ):
        return "classify"
    raise RuntimeError(f"Role {role.value} does not map to ultralytics CLI task")


def _resolve_ultralytics_data_arg(spec: TrainingRunSpec, task: str) -> str:
    """Resolve the `data=` argument expected by Ultralytics for a training run."""
    dataset_ref = Path(spec.derived_dataset_dir).expanduser()
    if task in {"detect", "obb"} and dataset_ref.is_dir():
        yaml_path = dataset_ref / "dataset.yaml"
        if yaml_path.exists():
            return str(yaml_path.resolve())
    return str(spec.derived_dataset_dir)


def build_ultralytics_command(spec: TrainingRunSpec, run_dir: str | Path) -> list[str]:
    """Build deterministic Ultralytics train command for a role."""

    task = _ultralytics_task_for_role(spec.role)
    run_dir = Path(run_dir).expanduser().resolve()
    data_arg = _resolve_ultralytics_data_arg(spec, task)

    args = [
        task,
        "train",
        f"data={data_arg}",
        f"model={spec.base_model}",
        f"epochs={int(spec.hyperparams.epochs)}",
        f"imgsz={int(spec.hyperparams.imgsz)}",
        f"batch={int(spec.hyperparams.batch)}",
        f"lr0={float(spec.hyperparams.lr0)}",
        f"patience={int(spec.hyperparams.patience)}",
        f"workers={int(spec.hyperparams.workers)}",
        f"project={str(run_dir.parent)}",
        f"name={run_dir.name}",
        "exist_ok=True",
        f"seed={int(spec.seed)}",
    ]
    if spec.device and spec.device != "auto":
        args.append(f"device={spec.device}")
    if spec.hyperparams.cache:
        args.append("cache=True")
    if spec.augmentation_profile.enabled and spec.augmentation_profile.args:
        for k, v in sorted(spec.augmentation_profile.args.items()):
            args.append(f"{k}={v}")
    if spec.resume_from:
        args.append("resume=True")

    yolo_exe = shutil.which("yolo")
    if yolo_exe:
        return [yolo_exe, *args]
    return [sys.executable, "-m", "ultralytics", *args]


def _pick_torch_device(requested: str) -> str:
    requested = str(requested or "auto").strip().lower()
    try:
        import torch
    except Exception:
        return "cpu"

    if requested not in {"", "auto"}:
        if requested.startswith("cuda") and torch.cuda.is_available():
            return requested
        if requested == "mps" and torch.backends.mps.is_available():
            return "mps"
        if requested == "cpu":
            return "cpu"

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_class_to_idx(dataset_dir: Path) -> dict[str, int]:
    """Build a stable class-name→index mapping across all classify splits.

    Using one shared mapping for train/val/test prevents label-index drift when a
    split is missing one or more classes.
    """
    names: set[str] = set()
    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        for cls_dir in split_dir.iterdir():
            if cls_dir.is_dir():
                names.add(cls_dir.name)
    ordered = sorted(names)
    return {name: idx for idx, name in enumerate(ordered)}


def _iter_classify_samples(
    dataset_dir: Path, split: str, class_to_idx: dict[str, int] | None = None
):
    """Yield ``(image_path, class_index)`` for all images in a classify split."""
    split_dir = dataset_dir / split
    if not split_dir.exists():
        return
    if class_to_idx is None:
        class_to_idx = _build_class_to_idx(dataset_dir)
    class_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
    for cls_dir in class_dirs:
        cls_idx = class_to_idx.get(cls_dir.name)
        if cls_idx is None:
            continue
        for img in sorted(cls_dir.rglob("*")):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                yield img, cls_idx


def _validate_tiny_samples(class_to_idx, train_samples):
    """Validate that train samples meet minimum requirements for tiny classify."""
    if len(train_samples) < 2:
        raise RuntimeError("Tiny classify training requires at least 2 train samples.")
    num_classes = len(class_to_idx)
    if num_classes < 2:
        raise RuntimeError("Need at least 2 classes in train split.")
    train_label_set = {int(lbl) for _, lbl in train_samples}
    if len(train_label_set) < 2:
        raise RuntimeError(
            "Need at least 2 classes represented in train split for tiny classify."
        )
    return num_classes


def _parse_tiny_rebalance_params(spec):
    """Extract rebalance mode, power, and label smoothing from spec."""
    rebalance_mode = (
        str(getattr(spec.tiny_params, "class_rebalance_mode", "none") or "none")
        .strip()
        .lower()
    )
    rebalance_power = float(
        max(0.0, getattr(spec.tiny_params, "class_rebalance_power", 1.0))
    )
    label_smoothing = float(
        min(0.4, max(0.0, getattr(spec.tiny_params, "label_smoothing", 0.0)))
    )
    return rebalance_mode, rebalance_power, label_smoothing


def _compute_class_weights(train_samples, num_classes, rebalance_mode, rebalance_power):
    """Compute inverse-frequency class weights for rebalancing."""
    class_counts = [0] * num_classes
    for _p, lbl in train_samples:
        if 0 <= int(lbl) < num_classes:
            class_counts[int(lbl)] += 1

    class_weight_values = [1.0] * num_classes
    if rebalance_mode in {"weighted_loss", "weighted_sampler", "both"}:
        max_count = max(class_counts) if class_counts else 1
        for idx in range(num_classes):
            count = max(1, class_counts[idx])
            class_weight_values[idx] = float(max_count / count) ** rebalance_power
        mean_w = sum(class_weight_values) / max(1, len(class_weight_values))
        if mean_w > 0:
            class_weight_values = [w / mean_w for w in class_weight_values]
    return class_weight_values


def _build_tiny_dataset_class(input_w, input_h):
    """Build and return a TinyDataset class closed over the input dimensions."""
    import cv2
    import torch
    from torch.utils.data import Dataset

    class TinyDataset(Dataset):
        """Image dataset that loads crops, applies optional augmentation, and normalizes for TinyClassifier training."""

        def __init__(self, items, augment=False, profile=None):
            self.items = items
            self.augment = augment
            self.profile = profile

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            path, label = self.items[idx]
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Could not read image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = _apply_tiny_augmentation(img, self.augment, self.profile)
            if img.shape[1] != input_w or img.shape[0] != input_h:
                img = cv2.resize(
                    img, (input_w, input_h), interpolation=cv2.INTER_LINEAR
                )
            x = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
            y = torch.tensor(label, dtype=torch.long)
            return x, y

    return TinyDataset


def _apply_tiny_augmentation(img, augment, profile):
    """Apply optional canonical-pose-safe augmentations to an image."""
    import cv2
    import numpy as np

    if not (augment and profile and profile.enabled):
        return img
    if profile.flipud > 0 and random.random() < profile.flipud:
        img = cv2.flip(img, 0)
    if profile.fliplr > 0 and random.random() < profile.fliplr:
        img = cv2.flip(img, 1)
    if profile.brightness > 0:
        factor = random.uniform(
            max(0.0, 1.0 - profile.brightness), 1.0 + profile.brightness
        )
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    if profile.contrast > 0:
        factor = random.uniform(
            max(0.0, 1.0 - profile.contrast), 1.0 + profile.contrast
        )
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(
            np.uint8
        )
    if profile.saturation > 0 or profile.hue > 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        if profile.saturation > 0:
            sat_factor = random.uniform(
                max(0.0, 1.0 - profile.saturation), 1.0 + profile.saturation
            )
            hsv[..., 1] = np.clip(hsv[..., 1] * sat_factor, 0, 255)
        if profile.hue > 0:
            hue_delta = random.uniform(-profile.hue, profile.hue) * 179.0
            hsv[..., 0] = (hsv[..., 0] + hue_delta) % 180.0
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if getattr(profile, "monochrome", False):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return img


def _create_tiny_data_loaders(
    TinyDataset, train_samples, val_samples, spec, rebalance_mode, class_weight_values
):
    """Create train and val DataLoaders with optional weighted sampling."""
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    train_sampler = None
    if rebalance_mode in {"weighted_sampler", "both"}:
        sample_weights = [class_weight_values[int(lbl)] for _p, lbl in train_samples]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
    train_loader = DataLoader(
        TinyDataset(train_samples, augment=True, profile=spec.augmentation_profile),
        batch_size=max(1, int(spec.tiny_params.batch)),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,
    )
    val_loader = (
        DataLoader(
            TinyDataset(val_samples, augment=False),
            batch_size=max(1, int(spec.tiny_params.batch)),
            shuffle=False,
            num_workers=0,
        )
        if val_samples
        else None
    )
    return train_loader, val_loader


def _run_tiny_training_loop(
    model,
    train_loader,
    val_loader,
    criterion,
    opt,
    device,
    epochs,
    patience,
    log_cb,
    progress_cb,
    should_cancel,
):
    """Run the training/validation loop.

    Returns (best_state, best_val_acc, history).
    """
    import time as _time

    best_val_acc = None
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history = []
    has_validation = bool(val_loader)
    _t0 = _time.monotonic()

    for epoch in range(epochs):
        if should_cancel and should_cancel():
            raise RuntimeError("Canceled")

        model.train()
        train_loss, train_n = 0.0, 0
        for xs, ys in train_loader:
            xs, ys = xs.to(device), ys.to(device)
            opt.zero_grad()
            loss = criterion(model(xs), ys)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * len(ys)
            train_n += len(ys)

        val_acc = _run_tiny_validation(model, val_loader, device)

        mean_loss = train_loss / max(1, train_n)
        history.append(
            {"epoch": epoch + 1, "train_loss": mean_loss, "val_acc": val_acc}
        )

        if has_validation:
            if best_val_acc is None or (
                val_acc is not None and val_acc >= best_val_acc
            ):
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                patience_counter = 0
                _safe_log(
                    log_cb,
                    f"  * new best val_acc={val_acc:.4f} at epoch {epoch + 1} — checkpoint saved",
                )
            else:
                patience_counter += 1
        else:
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        _safe_log(
            log_cb,
            (
                f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f} val_acc={val_acc:.4f}"
                if val_acc is not None
                else f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f} val_acc=n/a"
            ),
        )
        if progress_cb:
            progress_cb(epoch + 1, epochs)

        if has_validation and patience_counter >= patience:
            _safe_log(log_cb, f"Early stopping triggered at epoch {epoch + 1}")
            break

    _elapsed = _time.monotonic() - _t0
    _final_epoch = min(epoch + 1, epochs) if epochs > 0 else 0
    if has_validation and best_val_acc is not None:
        _safe_log(
            log_cb,
            f"Training loop done: best_val_acc={best_val_acc:.4f} @ epoch {best_epoch}/{_final_epoch}, "
            f"wall_time={_elapsed:.1f}s",
        )
    else:
        _safe_log(
            log_cb,
            f"Training loop done: {_final_epoch} epochs, wall_time={_elapsed:.1f}s",
        )

    return best_state, best_val_acc, history


def _run_tiny_validation(model, val_loader, device):
    """Evaluate model on validation set and return accuracy."""
    import torch

    if not val_loader:
        return None
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for xs, ys in val_loader:
            xs, ys = xs.to(device), ys.to(device)
            preds = model(xs).argmax(dim=1)
            correct += int((preds == ys).sum().item())
            total += len(ys)
    return correct / max(1, total)


def _save_tiny_checkpoint(
    *,
    model,
    save_path: str,
    class_names: list,
    input_size: tuple,
    monochrome: bool,
    hidden_layers: int,
    hidden_dim: int,
    dropout: float,
    best_val_acc,
    history,
) -> None:
    """Save a TinyClassifier checkpoint in the v2 classifier-artifact format.

    ``input_size`` is serialised as ``[H, W]``. ``class_names_per_factor`` is
    a length-1 list of the flat class list; ``class_names`` is kept for
    readability. ``schema_version`` is pinned at 2.

    ``history`` may be a list of per-epoch dicts (from the tiny training loop)
    or a plain dict; it is stored as-is.
    """
    import torch as _torch

    from hydra_suite.training.tiny_model import tiny_model_checkpoint_metadata

    h, w = int(input_size[0]), int(input_size[1])
    ckpt_dict = {
        "schema_version": 2,
        "arch": "tinyclassifier",
        "input_size": [h, w],
        "factor_names": ["flat"],
        "class_names_per_factor": [list(class_names)],
        "class_names": list(class_names),
        "num_classes": len(class_names),
        "monochrome": bool(monochrome),
        "hidden_layers": int(hidden_layers),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "best_val_acc": (float(best_val_acc) if best_val_acc is not None else None),
        "history": history if history is not None else [],
        "model_state_dict": model.state_dict(),
    }
    ckpt_dict.update(tiny_model_checkpoint_metadata(model))
    _torch.save(ckpt_dict, str(save_path))


def emit_yolo_multihead_manifest(
    *,
    manifest_path: str,
    factors: list[tuple[str, Path, list[str]]],
    input_size: tuple[int, int],
    monochrome: bool,
) -> Path:
    """Write a .multihead.json manifest for a multi-head YOLO bundle.

    ``factors`` is a list of ``(factor_name, pt_path, class_names)`` tuples.
    Per-factor .pt paths are stored relative to the manifest location.
    """
    import json as _json

    manifest_abs = Path(manifest_path).expanduser().resolve()
    manifest_abs.parent.mkdir(parents=True, exist_ok=True)

    factor_names: list[str] = []
    factor_entries: list[dict[str, Any]] = []
    for factor_name, pt_path, class_names in factors:
        pt_abs = Path(pt_path).expanduser().resolve()
        try:
            rel = pt_abs.relative_to(manifest_abs.parent).as_posix()
        except ValueError:
            rel = str(pt_abs)
        factor_names.append(str(factor_name))
        factor_entries.append(
            {
                "factor": str(factor_name),
                "path": rel,
                "class_names": [str(c) for c in class_names],
            }
        )

    payload = {
        "schema_version": 2,
        "kind": "yolo_multihead_bundle",
        "factor_names": factor_names,
        "factor_models": factor_entries,
        "input_size": [int(input_size[0]), int(input_size[1])],
        "monochrome": bool(monochrome),
    }
    manifest_abs.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_abs


def _try_onnx_export(model, ckpt_dict, out_ckpt, log_cb):
    """Attempt ONNX export alongside .pth; return path or None."""
    try:
        from hydra_suite.training.tiny_model import export_tiny_to_onnx

        _onnx_candidate = out_ckpt.with_suffix(".onnx")
        export_tiny_to_onnx(model, ckpt_dict, _onnx_candidate)
        _safe_log(log_cb, f"ONNX exported: {_onnx_candidate.name}")
        return _onnx_candidate
    except Exception as _onnx_exc:
        _safe_log(
            log_cb, f"ONNX export skipped ({type(_onnx_exc).__name__}: {_onnx_exc})"
        )
        return None


def _load_compatible_checkpoint_weights(
    model,
    checkpoint_path: str | Path,
    *,
    expected_arch: str = "",
    log_cb: LogCallback | None = None,
) -> dict:
    """Load compatible tensors from a prior checkpoint into a training model."""
    import torch

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(
            f"Starting checkpoint '{checkpoint_path}' does not contain model weights."
        )

    checkpoint_arch = str(ckpt.get("arch") or "").strip()
    if checkpoint_arch and expected_arch and checkpoint_arch != expected_arch:
        raise RuntimeError(
            "Starting checkpoint backbone does not match the selected training backbone: "
            f"checkpoint={checkpoint_arch}, selected={expected_arch}."
        )

    model_state = model.state_dict()
    compatible_state = {}
    skipped = 0
    for name, value in state_dict.items():
        target = model_state.get(name)
        if target is not None and getattr(target, "shape", None) == getattr(
            value, "shape", None
        ):
            compatible_state[name] = value
        else:
            skipped += 1

    if not compatible_state:
        raise RuntimeError(
            f"Starting checkpoint '{checkpoint_path}' is incompatible with the current model configuration."
        )

    model.load_state_dict(compatible_state, strict=False)
    _safe_log(
        log_cb,
        "Warm-started training from "
        f"{checkpoint_path.name}: loaded {len(compatible_state)} tensors"
        + (f", skipped {skipped} mismatched tensors." if skipped else "."),
    )
    return {"loaded": len(compatible_state), "skipped": skipped, "checkpoint": ckpt}


def _train_tiny_classify(
    spec: TrainingRunSpec,
    run_dir: Path,
    log_cb: LogCallback | None = None,
    progress_cb: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> dict:
    """Train a tiny N-class CNN classifier from an image-folder dataset."""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError(f"Tiny classify training requires torch/cv2: {exc}") from exc

    import time as _time

    _t0 = _time.monotonic()

    dataset_dir = Path(spec.derived_dataset_dir).expanduser().resolve()
    device = _pick_torch_device(spec.device)
    _safe_log(log_cb, f"Tiny classify device: {device}")

    class_to_idx = _build_class_to_idx(dataset_dir)
    train_samples = list(_iter_classify_samples(dataset_dir, "train", class_to_idx))
    val_samples = list(_iter_classify_samples(dataset_dir, "val", class_to_idx))
    num_classes = _validate_tiny_samples(class_to_idx, train_samples)

    _safe_log(
        log_cb,
        f"Dataset: {len(train_samples)} train, {len(val_samples)} val, "
        f"{num_classes} classes",
    )
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    for cls_idx in sorted(idx_to_class):
        n_train = sum(1 for _, lbl in train_samples if lbl == cls_idx)
        n_val = sum(1 for _, lbl in val_samples if lbl == cls_idx)
        _safe_log(
            log_cb,
            f"  class {cls_idx} ({idx_to_class[cls_idx]}): train={n_train}, val={n_val}",
        )

    input_w = int(spec.tiny_params.input_width)
    input_h = int(spec.tiny_params.input_height)
    rebalance_mode, rebalance_power, label_smoothing = _parse_tiny_rebalance_params(
        spec
    )
    class_weight_values = _compute_class_weights(
        train_samples, num_classes, rebalance_mode, rebalance_power
    )

    _safe_log(
        log_cb,
        "tiny classify options: "
        f"rebalance={rebalance_mode}, power={rebalance_power:.2f}, "
        f"label_smoothing={label_smoothing:.2f}",
    )

    TinyDataset = _build_tiny_dataset_class(input_w, input_h)

    from .tiny_model import _build_tiny_classifier_class

    _TinyClassifier = _build_tiny_classifier_class()

    class _TinyClassifierCompat(_TinyClassifier):
        """Thin wrapper accepting a params object instead of kwargs."""

        def __init__(self, n_classes, params):
            super().__init__(
                n_classes=n_classes,
                hidden_layers=params.hidden_layers,
                hidden_dim=params.hidden_dim,
                dropout=params.dropout,
            )

    model = _TinyClassifierCompat(num_classes, spec.tiny_params).to(device)
    _n_params = sum(p.numel() for p in model.parameters())
    _n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _safe_log(
        log_cb,
        f"Model: TinyClassifier — {_n_params:,} params ({_n_trainable:,} trainable)",
    )

    if spec.resume_from:
        _load_compatible_checkpoint_weights(
            model,
            spec.resume_from,
            expected_arch="tinyclassifier",
            log_cb=log_cb,
        )

    train_loader, val_loader = _create_tiny_data_loaders(
        TinyDataset,
        train_samples,
        val_samples,
        spec,
        rebalance_mode,
        class_weight_values,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec.tiny_params.lr),
        weight_decay=float(spec.tiny_params.weight_decay),
    )
    class_weight_tensor = None
    if rebalance_mode in {"weighted_loss", "both"}:
        class_weight_tensor = torch.as_tensor(
            class_weight_values, dtype=torch.float32, device=device
        )
    criterion = nn.CrossEntropyLoss(
        weight=class_weight_tensor,
        label_smoothing=label_smoothing,
    )

    epochs = max(1, int(spec.tiny_params.epochs))
    patience = max(1, int(spec.tiny_params.patience))

    best_state, best_val_acc, history = _run_tiny_training_loop(
        model,
        train_loader,
        val_loader,
        criterion,
        opt,
        device,
        epochs,
        patience,
        log_cb,
        progress_cb,
        should_cancel,
    )

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    _safe_log(log_cb, "Saving checkpoint and attempting ONNX export...")

    # Path derivation
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    sorted_class_items = sorted(class_to_idx.items(), key=lambda kv: int(kv[1]))
    _role_slug = spec.role.value.replace("classify_", "")
    _class_slug = "-".join(name for name, _idx in sorted_class_items)
    if len(_class_slug) > 48:
        _class_slug = f"{num_classes}cls"
    _run_stem = run_dir.parent.name
    _model_filename = f"classkit_{_role_slug}_{_class_slug}_{_run_stem}.pth"
    out_ckpt = weights_dir / _model_filename

    class_names_list = [name for name, _idx in sorted_class_items]

    _save_tiny_checkpoint(
        model=model,
        save_path=str(out_ckpt),
        class_names=class_names_list,
        input_size=(int(input_h), int(input_w)),
        monochrome=bool(getattr(spec.augmentation_profile, "monochrome", False)),
        hidden_layers=int(spec.tiny_params.hidden_layers),
        hidden_dim=int(spec.tiny_params.hidden_dim),
        dropout=float(spec.tiny_params.dropout),
        best_val_acc=best_val_acc,
        history=history,
    )

    # ONNX export
    _ckpt_for_onnx = {"input_size": [int(input_h), int(input_w)]}
    _onnx_path = _try_onnx_export(model, _ckpt_for_onnx, out_ckpt, log_cb)

    # Metrics JSON
    import json as _json

    metrics_path = run_dir / "tiny_metrics.json"
    metrics_path.write_text(
        _json.dumps(
            {
                "best_val_acc": (
                    float(best_val_acc) if best_val_acc is not None else None
                ),
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _safe_log(log_cb, f"Checkpoint saved: {out_ckpt.name}")

    _total_elapsed = _time.monotonic() - _t0
    _safe_log(
        log_cb,
        f"Tiny classify complete — total wall_time={_total_elapsed:.1f}s"
        + (f", best_val_acc={best_val_acc:.4f}" if best_val_acc is not None else ""),
    )

    return {
        "success": True,
        "artifact_path": str(out_ckpt),
        "onnx_path": str(_onnx_path) if _onnx_path is not None else "",
        "metrics_path": str(metrics_path),
        "best_val_acc": (float(best_val_acc) if best_val_acc is not None else None),
        "command": ["tiny_classify_inprocess"],
        "task": "tiny_classify",
    }


def _build_discriminative_param_groups(model, params):
    """Split model parameters into head and backbone groups with discriminative LR."""
    head_params, backbone_params = [], []
    head_names = {"classifier", "fc", "head", "heads"}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(h) for h in head_names):
            head_params.append(p)
        else:
            backbone_params.append(p)
    param_groups = [{"params": head_params, "lr": params.lr}]
    if backbone_params:
        param_groups.append(
            {"params": backbone_params, "lr": params.lr * params.backbone_lr_scale}
        )
    return param_groups


def _build_full_finetune_param_groups(model, params):
    """Train all currently-enabled parameters with a uniform learning rate."""
    params_list = [p for p in model.parameters() if p.requires_grad]
    return [{"params": params_list, "lr": params.lr}] if params_list else []


def _build_layerwise_lr_decay_param_groups(
    model, backbone, params, get_layer_groups_fn
):
    """Assign progressively smaller learning rates to earlier backbone groups."""
    head_names = {"classifier", "fc", "head", "heads"}
    head_params = []
    assigned_ids = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(name.startswith(prefix) for prefix in head_names):
            head_params.append(parameter)
            assigned_ids.add(id(parameter))

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": params.lr})

    decay = float(getattr(params, "layerwise_lr_decay", 0.75) or 0.75)
    decay = min(max(decay, 0.1), 1.0)
    groups = get_layer_groups_fn(model, backbone)
    for depth, group in enumerate(reversed(groups), start=1):
        group_params = []
        for parameter in group.parameters():
            if parameter.requires_grad and id(parameter) not in assigned_ids:
                group_params.append(parameter)
                assigned_ids.add(id(parameter))
        if group_params:
            param_groups.append(
                {"params": group_params, "lr": params.lr * (decay**depth)}
            )

    leftover = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in assigned_ids
    ]
    if leftover:
        param_groups.append(
            {"params": leftover, "lr": params.lr * (decay ** (len(groups) + 1))}
        )
    return param_groups


def _resolve_initial_trainable_layers(params, get_layer_groups_fn, model, backbone):
    """Map the requested fine-tuning strategy to an initial unfreeze depth."""
    method = str(getattr(params, "fine_tune_method", "head_only") or "head_only")
    if params.backbone == "tinyclassifier":
        return -1
    if method == "full_finetune":
        return -1
    if method == "head_only":
        return 0
    if method == "partial_unfreezing":
        total = len(get_layer_groups_fn(model, backbone))
        configured = int(getattr(params, "trainable_layers", 1) or 1)
        return max(1, min(total, configured))
    if method == "layerwise_lr_decay":
        return -1
    if method == "gradual_unfreezing":
        return 0
    return int(getattr(params, "trainable_layers", 0) or 0)


def _build_optimizer_for_fine_tune_strategy(
    model,
    backbone,
    params,
    torch_module,
    get_layer_groups_fn,
):
    """Build an AdamW optimizer matching the selected fine-tuning strategy."""
    method = str(getattr(params, "fine_tune_method", "head_only") or "head_only")
    if method == "full_finetune":
        param_groups = _build_full_finetune_param_groups(model, params)
    elif method == "layerwise_lr_decay":
        param_groups = _build_layerwise_lr_decay_param_groups(
            model, backbone, params, get_layer_groups_fn
        )
    else:
        param_groups = _build_discriminative_param_groups(model, params)
    return torch_module.optim.AdamW(param_groups, weight_decay=params.weight_decay)


def _run_torchvision_training_loop(
    model,
    backbone,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    params,
    save_torchvision_checkpoint_fn,
    freeze_backbone_fn,
    get_layer_groups_fn,
    best_ckpt_path,
    class_names,
    checkpoint_extra_meta,
    log_cb,
    progress_cb,
    should_cancel,
):
    """Run training/validation epochs for a torchvision model.

    Returns (best_val_acc, history).
    """
    import time as _time

    import torch

    sz = params.input_size
    best_val_acc = None
    best_epoch = 0
    patience_count = 0
    history: dict = {"train_loss": [], "val_acc": []}
    has_validation = bool(val_loader)
    current_unfrozen_groups = 0
    _t0 = _time.monotonic()
    total_groups = 0
    if backbone != "tinyclassifier":
        try:
            total_groups = len(get_layer_groups_fn(model, backbone))
        except Exception:
            total_groups = 0

    for epoch in range(params.epochs):
        if should_cancel and should_cancel():
            _safe_log(log_cb, "Training canceled.")
            break

        if (
            backbone != "tinyclassifier"
            and str(getattr(params, "fine_tune_method", "head_only"))
            == "gradual_unfreezing"
            and total_groups > 0
        ):
            interval = max(1, int(getattr(params, "gradual_unfreeze_interval", 5) or 5))
            desired_groups = min(total_groups, epoch // interval)
            if desired_groups != current_unfrozen_groups:
                freeze_backbone_fn(model, backbone, desired_groups)
                optimizer = _build_optimizer_for_fine_tune_strategy(
                    model,
                    backbone,
                    params,
                    torch,
                    get_layer_groups_fn,
                )
                current_unfrozen_groups = desired_groups
                _safe_log(
                    log_cb,
                    f"Gradual unfreezing: training head + last {desired_groups} backbone groups.",
                )

        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(train_loader), 1)
        history["train_loss"].append(avg_loss)

        val_acc = None
        if has_validation:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    preds = model(batch_x).argmax(dim=1)
                    correct += (preds == batch_y).sum().item()
                    total += len(batch_y)
            val_acc = correct / max(total, 1)
        history["val_acc"].append(val_acc)

        _safe_log(
            log_cb,
            (
                f"Epoch {epoch + 1}/{params.epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}"
                if val_acc is not None
                else f"Epoch {epoch + 1}/{params.epochs}  loss={avg_loss:.4f}  val_acc=n/a"
            ),
        )
        if progress_cb:
            progress_cb(epoch + 1, params.epochs)

        if has_validation and val_acc is not None:
            if best_val_acc is None or val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_count = 0
                save_torchvision_checkpoint_fn(
                    model=model,
                    backbone=backbone,
                    class_names=class_names,
                    factor_names=[],
                    input_size=(sz, sz),
                    best_val_acc=best_val_acc,
                    history=history,
                    trainable_layers=params.trainable_layers,
                    backbone_lr_scale=params.backbone_lr_scale,
                    monochrome=bool(checkpoint_extra_meta.get("monochrome", False)),
                    extra_meta=checkpoint_extra_meta,
                    path=best_ckpt_path,
                )
                _safe_log(
                    log_cb,
                    f"  * new best val_acc={val_acc:.4f} at epoch {epoch + 1} — checkpoint saved",
                )
            else:
                patience_count += 1
                if patience_count >= params.patience:
                    _safe_log(log_cb, f"Early stopping at epoch {epoch + 1}.")
                    break

    if not has_validation and history["train_loss"]:
        save_torchvision_checkpoint_fn(
            model=model,
            backbone=backbone,
            class_names=class_names,
            factor_names=[],
            input_size=(sz, sz),
            best_val_acc=None,
            history=history,
            trainable_layers=params.trainable_layers,
            backbone_lr_scale=params.backbone_lr_scale,
            monochrome=bool(checkpoint_extra_meta.get("monochrome", False)),
            extra_meta=checkpoint_extra_meta,
            path=best_ckpt_path,
        )
        _safe_log(log_cb, "Final checkpoint saved (no validation).")

    _elapsed = _time.monotonic() - _t0
    _final_epoch = min(epoch + 1, params.epochs) if params.epochs > 0 else 0
    if has_validation and best_val_acc is not None:
        _safe_log(
            log_cb,
            f"Training loop done: best_val_acc={best_val_acc:.4f} @ epoch {best_epoch}/{_final_epoch}, "
            f"wall_time={_elapsed:.1f}s",
        )
    else:
        _safe_log(
            log_cb,
            f"Training loop done: {_final_epoch} epochs, wall_time={_elapsed:.1f}s",
        )

    return best_val_acc, history


def _train_custom_classify(
    spec: "TrainingRunSpec",
    run_dir: Path,
    log_cb=None,
    progress_cb=None,
    should_cancel=None,
) -> dict:
    """Train a Custom CNN classifier (TinyClassifier or torchvision backbone).

    If backbone == 'tinyclassifier', delegates entirely to _train_tiny_classify().
    Otherwise trains a pretrained torchvision model with discriminative LR.
    """
    from .torchvision_model import (
        build_torchvision_classifier,
        export_torchvision_to_onnx,
        freeze_backbone,
        get_classifier_normalization_stats,
        get_layer_groups,
        load_torchvision_classifier,
        save_torchvision_checkpoint,
    )

    params: CustomCNNParams = spec.custom_params or CustomCNNParams()

    if params.backbone == "tinyclassifier":
        translated_spec = dataclasses.replace(
            spec,
            tiny_params=dataclasses.replace(
                spec.tiny_params,
                epochs=int(params.epochs),
                batch=int(params.batch),
                lr=float(params.lr),
                weight_decay=float(params.weight_decay),
                input_width=int(params.input_width),
                input_height=int(params.input_height),
                tiny_preset=str(params.tiny_preset or "medium"),
                hidden_layers=int(params.hidden_layers),
                hidden_dim=int(params.hidden_dim),
                dropout=float(params.dropout),
                patience=int(params.patience),
                class_rebalance_mode=str(params.class_rebalance_mode),
                class_rebalance_power=float(params.class_rebalance_power),
                label_smoothing=float(params.label_smoothing),
            ),
        )
        return _train_tiny_classify(
            translated_spec,
            run_dir,
            log_cb,
            progress_cb,
            should_cancel,
        )

    # --- Torchvision training path ---
    import json
    import time as _time

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    _t0 = _time.monotonic()

    run_dir = Path(run_dir)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(spec.derived_dataset_dir)
    sz = params.input_size

    profile = spec.augmentation_profile
    mean, std = get_classifier_normalization_stats(
        monochrome=bool(getattr(profile, "monochrome", False))
    )
    train_transforms = [transforms.Resize((sz, sz))]
    if profile.fliplr > 0:
        train_transforms.append(transforms.RandomHorizontalFlip(p=profile.fliplr))
    if profile.flipud > 0:
        train_transforms.append(transforms.RandomVerticalFlip(p=profile.flipud))
    if (
        profile.brightness > 0
        or profile.contrast > 0
        or getattr(profile, "saturation", 0.0) > 0
        or getattr(profile, "hue", 0.0) > 0
    ):
        train_transforms.append(
            transforms.ColorJitter(
                brightness=float(profile.brightness),
                contrast=float(profile.contrast),
                saturation=float(getattr(profile, "saturation", 0.0)),
                hue=float(getattr(profile, "hue", 0.0)),
            )
        )
    if getattr(profile, "monochrome", False):
        train_transforms.append(transforms.Grayscale(num_output_channels=3))
    train_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_tf = transforms.Compose(train_transforms)
    val_tf = transforms.Compose(
        [
            transforms.Resize((sz, sz)),
            *(
                [transforms.Grayscale(num_output_channels=3)]
                if getattr(profile, "monochrome", False)
                else []
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Build a shared class→index mapping across all splits so that indices
    # are consistent even when a class is absent from the validation split.
    # Without this, ImageFolder assigns indices independently per split,
    # silently corrupting validation labels when a class is missing from val.
    shared_class_to_idx = _build_class_to_idx(dataset_dir)
    class_names = sorted(shared_class_to_idx.keys())

    train_ds = datasets.ImageFolder(str(dataset_dir / "train"), transform=train_tf)
    val_dir = dataset_dir / "val"
    has_validation = val_dir.exists() and any(
        path.is_file() for path in val_dir.rglob("*")
    )
    val_ds = (
        datasets.ImageFolder(str(val_dir), transform=val_tf) if has_validation else None
    )

    for ds in [train_ds] + ([val_ds] if val_ds else []):
        if ds.class_to_idx != shared_class_to_idx:
            old_to_name = {v: k for k, v in ds.class_to_idx.items()}
            remap = {
                old_idx: shared_class_to_idx[name]
                for old_idx, name in old_to_name.items()
            }
            ds.samples = [(p, remap[t]) for p, t in ds.samples]
            ds.targets = [remap[t] for t in ds.targets]
            ds.imgs = ds.samples
            ds.class_to_idx = shared_class_to_idx
            ds.classes = class_names
    checkpoint_extra_meta = {
        "fine_tune_method": getattr(params, "fine_tune_method", "head_only"),
        "layerwise_lr_decay": getattr(params, "layerwise_lr_decay", 0.75),
        "gradual_unfreeze_interval": getattr(params, "gradual_unfreeze_interval", 5),
        "monochrome": bool(getattr(profile, "monochrome", False)),
    }

    _n_train = len(train_ds)
    _n_val = len(val_ds) if val_ds else 0
    _safe_log(
        log_cb,
        f"Dataset: {_n_train} train, {_n_val} val, {len(class_names)} classes",
    )
    from collections import Counter as _Counter

    _train_dist = _Counter(train_ds.targets)
    _val_dist = _Counter(val_ds.targets) if val_ds else {}
    for cls_name in class_names:
        cls_idx = shared_class_to_idx[cls_name]
        _safe_log(
            log_cb,
            f"  class {cls_idx} ({cls_name}): "
            f"train={_train_dist.get(cls_idx, 0)}, val={_val_dist.get(cls_idx, 0)}",
        )

    if not has_validation:
        _safe_log(
            log_cb,
            "No validation split found; training without held-out validation.",
        )

    rebalance_mode = (
        str(getattr(params, "class_rebalance_mode", "none") or "none").strip().lower()
    )
    rebalance_power = float(max(0.0, getattr(params, "class_rebalance_power", 1.0)))

    class_weight_values = _compute_class_weights(
        [(p, lbl) for (p, _), lbl in zip(train_ds.imgs, train_ds.targets)],
        len(class_names),
        rebalance_mode,
        rebalance_power,
    )

    from torch.utils.data import WeightedRandomSampler

    train_sampler = None
    if rebalance_mode in {"weighted_sampler", "both"}:
        sample_weights = [class_weight_values[lbl] for lbl in train_ds.targets]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=params.batch,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=params.batch,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        if val_ds is not None
        else None
    )

    device = _pick_torch_device(spec.device)
    _safe_log(log_cb, f"Custom CNN device: {device}, backbone: {params.backbone}")
    model = build_torchvision_classifier(
        params.backbone,
        len(class_names),
        -1,
        tiny_preset=str(getattr(params, "tiny_preset", "medium") or "medium"),
        hidden_layers=params.hidden_layers,
        hidden_dim=params.hidden_dim,
        dropout=params.dropout,
        input_width=params.input_width,
        input_height=params.input_height,
        input_size=params.input_size,
    )
    _n_params = sum(p.numel() for p in model.parameters())
    _n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _safe_log(
        log_cb,
        f"Model: {params.backbone} — {_n_params:,} params ({_n_trainable:,} trainable)",
    )
    if spec.resume_from:
        _load_compatible_checkpoint_weights(
            model,
            spec.resume_from,
            expected_arch=params.backbone,
            log_cb=log_cb,
        )
    initial_trainable_layers = _resolve_initial_trainable_layers(
        params,
        get_layer_groups,
        model,
        params.backbone,
    )
    if initial_trainable_layers != -1:
        freeze_backbone(model, params.backbone, initial_trainable_layers)
    model.to(device)

    optimizer = _build_optimizer_for_fine_tune_strategy(
        model,
        params.backbone,
        params,
        torch,
        get_layer_groups,
    )
    criterion = nn.CrossEntropyLoss(
        weight=(
            torch.tensor(class_weight_values, dtype=torch.float).to(device)
            if rebalance_mode in {"weighted_loss", "both"}
            else None
        ),
        label_smoothing=params.label_smoothing,
    )

    best_ckpt_path = (
        weights_dir / f"classkit_custom_{params.backbone}_{len(class_names)}cls.pth"
    )

    best_val_acc, history = _run_torchvision_training_loop(
        model,
        params.backbone,
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
    )

    metrics_path = run_dir / "custom_metrics.json"
    metrics_path.write_text(
        json.dumps({"best_val_acc": best_val_acc, "history": history}, indent=2)
    )

    if not best_ckpt_path.exists():
        _safe_log(
            log_cb, "No checkpoint saved (training canceled or val acc never improved)."
        )
        return {
            "success": False,
            "artifact_path": "",
            "onnx_path": "",
            "metrics_path": str(metrics_path),
            "best_val_acc": best_val_acc,
            "command": ["custom_classify_inprocess"],
            "task": "custom_classify",
        }

    best_model, best_ckpt = load_torchvision_classifier(
        str(best_ckpt_path), device="cpu"
    )
    onnx_path = best_ckpt_path.with_suffix(".onnx")
    _safe_log(log_cb, "Exporting best checkpoint to ONNX...")
    try:
        export_torchvision_to_onnx(best_model, best_ckpt, onnx_path)
        _safe_log(log_cb, f"ONNX exported: {onnx_path.name}")
    except Exception as _onnx_exc:
        _safe_log(
            log_cb,
            f"ONNX export failed ({type(_onnx_exc).__name__}: {_onnx_exc})",
        )
        onnx_path = Path("")

    _total_elapsed = _time.monotonic() - _t0
    _safe_log(
        log_cb,
        f"Custom CNN training complete — total wall_time={_total_elapsed:.1f}s"
        + (f", best_val_acc={best_val_acc:.4f}" if best_val_acc is not None else ""),
    )

    return {
        "success": True,
        "artifact_path": str(best_ckpt_path),
        "onnx_path": str(onnx_path) if str(onnx_path) else "",
        "metrics_path": str(metrics_path),
        "best_val_acc": best_val_acc,
        "command": ["custom_classify_inprocess"],
        "task": "custom_classify",
    }


_CUSTOM_CLASSIFY_ROLES = {
    TrainingRole.CLASSIFY_FLAT_CUSTOM,
    TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM,
    TrainingRole.CLASSIFY_FLAT_TINY,
    TrainingRole.CLASSIFY_MULTIHEAD_TINY,
}


def _cancel_subprocess(proc, command):
    """Terminate or kill a subprocess and return a cancellation result dict."""
    try:
        proc.terminate()
    except Exception:
        pass
    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass
    return {
        "success": False,
        "canceled": True,
        "artifact_path": "",
        "metrics_path": "",
        "command": command,
    }


def _stream_ultralytics_output(proc, log_cb, progress_cb, should_cancel, command):
    """Stream subprocess output, parse progress, handle cancellation.

    Returns a cancellation result dict if canceled, otherwise None.
    """
    assert proc.stdout is not None
    ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

    def _extract_progress(msg: str) -> tuple[int, int] | None:
        progress_patterns = (
            re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE),
            re.compile(r"Epoch\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE),
            re.compile(r"^\s*(\d+)\s*/\s*(\d+)(?:\s|$)"),
            re.compile(
                r"\bepoch\s*[=:]\s*(\d+)\b.*\btotal\s*[=:]\s*(\d+)\b", re.IGNORECASE
            ),
        )
        for regex in progress_patterns:
            match = regex.search(msg)
            if match:
                return int(match.group(1)), int(match.group(2))
        return None

    for line in proc.stdout:
        if should_cancel and should_cancel():
            return _cancel_subprocess(proc, command)

        msg = ansi_re.sub("", line).rstrip()
        if msg:
            _safe_log(log_cb, msg)
        parsed_progress = _extract_progress(msg)
        if parsed_progress is not None and progress_cb is not None:
            try:
                progress_cb(*parsed_progress)
            except Exception as exc:
                _safe_log(log_cb, f"Progress callback failed: {exc}")
    return None


def run_training(
    spec: TrainingRunSpec,
    run_dir: str | Path,
    *,
    log_cb: LogCallback | None = None,
    progress_cb: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> dict:
    """Run one role training job and return structured result."""

    run_dir = Path(run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if spec.role in _CUSTOM_CLASSIFY_ROLES:
        if spec.custom_params is None:
            spec = dataclasses.replace(
                spec, custom_params=CustomCNNParams(backbone="tinyclassifier")
            )
        return _train_custom_classify(
            spec,
            run_dir,
            log_cb=log_cb,
            progress_cb=progress_cb,
            should_cancel=should_cancel,
        )

    command = build_ultralytics_command(spec, run_dir)
    _safe_log(log_cb, "Running: " + " ".join(command))

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    cancel_result = _stream_ultralytics_output(
        proc, log_cb, progress_cb, should_cancel, command
    )
    if cancel_result is not None:
        return cancel_result

    rc = proc.wait()
    if rc != 0:
        _safe_log(log_cb, f"YOLO process exited with code {rc}")
        return {
            "success": False,
            "exit_code": int(rc),
            "artifact_path": "",
            "metrics_path": "",
            "command": command,
        }

    weights_dir = run_dir / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    artifact = best if best.exists() else last if last.exists() else None
    if artifact is not None:
        _safe_log(log_cb, f"YOLO artifact selected: {artifact.name}")
    else:
        _safe_log(log_cb, "WARNING: No YOLO checkpoint found in weights directory")

    metrics_path = run_dir / "results.csv"
    best_val_acc = _extract_best_val_acc_from_results_csv(metrics_path)
    if best_val_acc is not None:
        _safe_log(log_cb, f"YOLO best val accuracy: {best_val_acc:.4f}")
    return {
        "success": artifact is not None,
        "artifact_path": str(artifact) if artifact is not None else "",
        "metrics_path": str(metrics_path) if metrics_path.exists() else "",
        "best_val_acc": best_val_acc,
        "command": command,
        "task": _ultralytics_task_for_role(spec.role),
    }
