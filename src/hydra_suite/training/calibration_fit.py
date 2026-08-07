"""Fit temperature-scaling calibration from a validation loader (Training layer).

Imports the calibration math from Core (allowed: Training -> Core). Produces one
temperature per factor plus ECE before/after and a model-weight signature, for
persistence into the model artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from hydra_suite.core.individual.identity.calibration import (
    expected_calibration_error,
    fit_temperature,
    model_weight_signature,
)


@dataclass
class CalibrationResult:
    temperatures: list[float]
    signature: str
    ece_before: list[float]
    ece_after: list[float]


def _softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@torch.no_grad()
def _collect(model, val_loader, device, split_logits, num_factors):
    model.eval()
    per_factor_logits: list[list[np.ndarray]] = [[] for _ in range(num_factors)]
    per_factor_labels: list[list[np.ndarray]] = [[] for _ in range(num_factors)]
    for batch in val_loader:
        xs, ys = batch[0].to(device), batch[1]
        out = model(xs)
        parts = split_logits(out) if split_logits is not None else [out]
        for k in range(num_factors):
            per_factor_logits[k].append(parts[k].detach().cpu().numpy())
            yk = ys if ys.ndim == 1 else ys[:, k]
            per_factor_labels[k].append(yk.detach().cpu().numpy())
    return per_factor_logits, per_factor_labels


def fit_calibration_from_val(
    model, val_loader, device: str, *, split_logits=None, num_factors: int = 1
) -> CalibrationResult:
    logits_by_f, labels_by_f = _collect(
        model, val_loader, device, split_logits, num_factors
    )
    temps, ece_b, ece_a = [], [], []
    for k in range(num_factors):
        logits = np.concatenate(logits_by_f[k], axis=0)
        labels = np.concatenate(labels_by_f[k], axis=0)
        ece_b.append(expected_calibration_error(_softmax_np(logits), labels))
        t = fit_temperature(logits, labels)
        temps.append(t)
        ece_a.append(expected_calibration_error(_softmax_np(logits / t), labels))
    sig = model_weight_signature(model.state_dict())
    return CalibrationResult(
        temperatures=temps, signature=sig, ece_before=ece_b, ece_after=ece_a
    )


def _build_recalibration_val_loader(model_path: str, val_dir: str, raw_ckpt: dict):
    """Build a val ``DataLoader`` over ``val_dir`` mirroring the training-side
    transform for the artifact's arch, plus a ``split_logits`` callable for
    multi-head checkpoints (``None`` for flat checkpoints).

    Reuses ``CanonicalFitTransform``/``cv2_bgr_loader`` (the one shared
    train/inference resize+letterbox path) and
    ``get_classifier_normalization_stats`` so the fitted temperature is on the
    same input distribution training produced. ``val_dir`` must be an
    ImageFolder-style directory (``val_dir/<class_name>/*.jpg``) for flat
    checkpoints, or a composite-folder directory
    (``val_dir/<f0>__<f1>.../*.jpg``) for multi-head shared-trunk checkpoints.
    """
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    from hydra_suite.training.canonical_transform import (
        CanonicalFitTransform,
        bgr_to_rgb_pil,
        cv2_bgr_loader,
    )
    from hydra_suite.training.torchvision_model import (
        get_classifier_normalization_stats,
    )

    arch = str(raw_ckpt.get("arch", "tinyclassifier"))
    factor_names = list(raw_ckpt.get("factor_names") or ["flat"])
    num_factors = len(factor_names)
    monochrome = bool(raw_ckpt.get("monochrome", False))
    input_size = raw_ckpt.get("input_size") or [64, 128]
    h, w = int(input_size[0]), int(input_size[1])

    if arch == "tinyclassifier":
        # TinyDataset (runner.py) normalizes by /255 only -- no ImageNet mean/std.
        val_tf = transforms.Compose(
            [
                CanonicalFitTransform((h, w)),
                transforms.Lambda(bgr_to_rgb_pil),
                transforms.ToTensor(),
            ]
        )
    else:
        mean, std = get_classifier_normalization_stats(monochrome=monochrome)
        val_tf = transforms.Compose(
            [CanonicalFitTransform((h, w)), transforms.Lambda(bgr_to_rgb_pil)]
            + ([transforms.Grayscale(num_output_channels=3)] if monochrome else [])
            + [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    split_logits = None
    if num_factors > 1:
        from hydra_suite.training.multihead_dataset import MultiFactorImageFolder

        cnpf = raw_ckpt.get("class_names_per_factor") or []
        val_ds = MultiFactorImageFolder(
            str(val_dir),
            class_names_per_factor=cnpf,
            delimiter="__",
            transform=val_tf,
        )
        factor_widths = [len(c) for c in cnpf]

        def split_logits(logits):
            out = []
            offset = 0
            for width in factor_widths:
                out.append(logits[:, offset : offset + width])
                offset += width
            return out

    else:
        val_ds = datasets.ImageFolder(
            str(val_dir), transform=val_tf, loader=cv2_bgr_loader
        )
        # ImageFolder indexes classes by sorted folder name, independent of the
        # checkpoint's stored class order. If val_dir's class set/order doesn't
        # match the checkpoint's, labels silently misalign against the model's
        # logit order and calibration fits a WRONG temperature with no error.
        # class_names_per_factor[0] is authoritative for flat checkpoints (see
        # save_torchvision_checkpoint); fall back to "class_names" if absent.
        cnpf_flat = raw_ckpt.get("class_names_per_factor") or []
        stored_classes = list(
            cnpf_flat[0] if cnpf_flat else raw_ckpt.get("class_names") or []
        )
        if stored_classes and list(val_ds.classes) != stored_classes:
            raise ValueError(
                f"{val_dir!r}: val_dir classes {list(val_ds.classes)!r} do not "
                f"match the checkpoint's class order {stored_classes!r} -- "
                "torchvision.datasets.ImageFolder indexes classes by sorted "
                "folder name, so a mismatched/missing class here would "
                "silently misalign labels against the model's logits and fit "
                "a wrong calibration temperature. Fix val_dir to contain "
                "exactly the checkpoint's classes."
            )

    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    return val_loader, num_factors, split_logits


def _rebuild_model_for_recalibration(model_path: str, raw_ckpt: dict, device: str):
    """Rebuild the model from a checkpoint for calibration refitting only.

    Dispatches on ``arch``: ``"tinyclassifier"`` reconstructs via
    ``tiny_model.load_tiny_classifier`` (infers hidden dims from the state
    dict); everything else (torchvision flat or multi-head shared-trunk)
    reconstructs via ``torchvision_model.load_torchvision_classifier``. Both
    return the model already ``.eval()``-ed with weights loaded, so
    calibration fitting never mutates ``model_state_dict``.
    """
    arch = str(raw_ckpt.get("arch", "tinyclassifier"))
    if arch == "tinyclassifier":
        from hydra_suite.training.tiny_model import load_tiny_classifier

        model, _ = load_tiny_classifier(model_path, device=device)
    else:
        from hydra_suite.training.torchvision_model import load_torchvision_classifier

        model, _ = load_torchvision_classifier(model_path, device=device)
    model.eval()
    return model


def _rewrite_artifact_calibration(model_path: str, result: CalibrationResult) -> None:
    """Rewrite calibration_temperature/signature/ece in place on the artifact.

    Only the three calibration keys change -- ``model_state_dict`` (and every
    other checkpoint field) round-trips untouched. If a ``.v2meta.json``
    sidecar exists next to the model (the YOLO flat-export convention), its
    calibration fields are updated too.
    """
    path = Path(model_path)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    ckpt["calibration_temperature"] = list(result.temperatures)
    ckpt["calibration_signature"] = result.signature
    ckpt["calibration_ece"] = list(result.ece_after)
    torch.save(ckpt, str(path))

    sidecar_path = path.with_suffix(".v2meta.json")
    if sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            sidecar = None
        if isinstance(sidecar, dict):
            sidecar["calibration_temperature"] = list(result.temperatures)
            sidecar["calibration_signature"] = result.signature
            sidecar["calibration_ece"] = list(result.ece_after)
            sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")


def recalibrate_artifact(model_path: str, val_dir: str) -> CalibrationResult:
    """Rebuild the model from ``model_path``, refit calibration on ``val_dir``,
    and rewrite the artifact's calibration metadata in place.

    ``val_dir`` is an ImageFolder-style directory (``val_dir/<class>/*.jpg``)
    for flat checkpoints, or a composite-folder directory for multi-head
    shared-trunk checkpoints (see ``MultiFactorImageFolder``). Only
    ``calibration_temperature``/``calibration_signature``/``calibration_ece``
    are rewritten -- model weights are unchanged.

    Supports ``.pth``/``.pt`` checkpoints (tiny, torchvision-flat, and
    torchvision multi-head shared-trunk). ``.multihead.json`` YOLO bundles
    (a manifest referencing several independent per-factor models, not a
    single artifact with weights) are not supported here -- recalibrate each
    factor's ``.pt`` individually.
    """
    path = Path(model_path)
    if path.name.lower().endswith(".multihead.json"):
        raise NotImplementedError(
            f"{model_path!r}: recalibrate_artifact does not support "
            ".multihead.json bundles (a manifest referencing several "
            "independent per-factor models) -- recalibrate each factor's "
            ".pt individually."
        )

    device = "cpu"
    raw_ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(raw_ckpt, dict):
        raise ValueError(f"{model_path!r}: expected a checkpoint dict")

    model = _rebuild_model_for_recalibration(model_path, raw_ckpt, device)
    val_loader, num_factors, split_logits = _build_recalibration_val_loader(
        model_path, val_dir, raw_ckpt
    )

    result = fit_calibration_from_val(
        model, val_loader, device, split_logits=split_logits, num_factors=num_factors
    )
    _rewrite_artifact_calibration(model_path, result)
    return result
