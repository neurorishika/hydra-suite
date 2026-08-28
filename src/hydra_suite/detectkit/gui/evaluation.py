"""Reusable DetectKit dataset-analysis and quick-test helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QMessageBox

if TYPE_CHECKING:
    from .models import DetectKitProject

logger = logging.getLogger(__name__)


def build_dataset_analysis_report(project: "DetectKitProject") -> tuple[str, list[str]]:
    """Return a merged dataset-analysis report and any warnings."""
    sources = project.sources
    if not sources:
        return "No dataset sources configured.", []

    try:
        from hydra_suite.training.dataset_inspector import (
            DatasetInspection,
            analyze_obb_sizes,
            format_size_analysis,
            inspect_obb_or_detect_dataset,
        )
    except ImportError:
        return (
            "Dataset inspector not available. Install training dependencies.",
            [],
        )

    merged = DatasetInspection(root_dir="(merged)")
    for src in sources:
        if not src.path:
            continue
        try:
            inspection = inspect_obb_or_detect_dataset(src.path)
        except Exception as exc:
            logger.warning("Failed to inspect %s: %s", src.path, exc)
            continue
        for split_name, items in inspection.splits.items():
            merged.splits.setdefault(split_name, []).extend(items)
        merged.class_names.update(inspection.class_names)

    if not any(merged.splits.values()):
        return "No valid dataset items found in the configured sources.", []

    stats = analyze_obb_sizes(
        merged,
        pad_ratio=project.crop_pad_ratio,
        min_crop_size_px=project.min_crop_size_px,
        enforce_square=project.enforce_square,
    )

    report_seq, warnings_seq = format_size_analysis(
        stats,
        training_imgsz=project.imgsz_seq_crop_obb,
        pipeline_mode="crop",
    )
    report_direct, warnings_direct = format_size_analysis(
        stats,
        training_imgsz=project.imgsz_obb_direct,
        pipeline_mode="full_image",
    )

    lines = [
        "=== Seq Crop OBB Pipeline ===",
        f"(imgsz = {project.imgsz_seq_crop_obb})",
        "",
        report_seq,
    ]
    if warnings_seq:
        lines += ["", "WARNINGS:"] + [f"  ! {warning}" for warning in warnings_seq]

    lines += [
        "",
        "=== OBB Direct Pipeline ===",
        f"(imgsz = {project.imgsz_obb_direct})",
        "",
        report_direct,
    ]
    if warnings_direct:
        lines += ["", "WARNINGS:"] + [f"  ! {warning}" for warning in warnings_direct]

    return "\n".join(lines), warnings_seq + warnings_direct


def open_quick_test_dialog(
    project: "DetectKitProject",
    *,
    parent=None,
) -> bool:
    """Open the shared quick-test dialog for the active DetectKit model."""
    model_path = str(project.active_model_path or "").strip()
    if not model_path or not Path(model_path).exists():
        QMessageBox.information(
            parent,
            "Quick Test",
            "No active model found.\n\n"
            "Run training first, or select a model from Run History.",
        )
        return False

    dataset_dir = project.sources[0].path if project.sources else ""

    from .project import detectkit_resolve_inference_models

    try:
        kind, primary, secondary = detectkit_resolve_inference_models(
            project, model_path
        )
    except RuntimeError as exc:
        QMessageBox.information(parent, "Quick Test", str(exc))
        return False

    role = kind
    quick_test_model_path = primary
    detect_model_path = ""
    if kind == "sequential":
        role = "seq_crop_obb"
        quick_test_model_path = str(secondary or "")
        detect_model_path = primary
    elif kind == "sequential_segment":
        role = "seq_crop_segment"
        quick_test_model_path = str(secondary or "")
        detect_model_path = primary

    imgsz = {
        "obb_direct": int(project.imgsz_obb_direct),
        "detect_direct": int(project.imgsz_detect_direct),
        "segment_direct": int(project.imgsz_segment_direct),
        "seq_crop_obb": int(project.imgsz_seq_crop_obb),
        "seq_crop_segment": int(project.imgsz_seq_crop_segment),
    }.get(role, int(project.imgsz_obb_direct))

    try:
        from hydra_suite.trackerkit.gui.dialogs.model_test_dialog import (
            ModelTestDialog,
            training_device_to_compute_runtime,
        )
    except ImportError:
        QMessageBox.information(
            parent,
            "Not Available",
            "Model test dialog is not available.",
        )
        return False

    dialog = ModelTestDialog(
        model_path=quick_test_model_path,
        role=role,
        dataset_dir=dataset_dir,
        compute_runtime=training_device_to_compute_runtime(project.device or "cpu"),
        imgsz=imgsz,
        crop_pad_ratio=project.crop_pad_ratio,
        min_crop_size_px=project.min_crop_size_px,
        enforce_square=project.enforce_square,
        detect_model_path=detect_model_path,
        parent=parent,
    )
    dialog.open()
    return True
