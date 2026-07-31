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

    try:
        from .project import (
            detectkit_latest_model_path_for_role,
            detectkit_training_history_entry_for_model_path,
        )
    except ImportError:
        detectkit_latest_model_path_for_role = None
        detectkit_training_history_entry_for_model_path = None

    entry = (
        detectkit_training_history_entry_for_model_path(project, model_path)
        if detectkit_training_history_entry_for_model_path is not None
        else None
    )
    role = (
        str((entry or {}).get("role") or "obb_direct").strip().lower() or "obb_direct"
    )
    if role == "seq_detect":
        QMessageBox.information(
            parent,
            "Quick Test",
            "The selected model is a sequence-detect stage-1 checkpoint. Quick Test supports OBB direct models and sequence crop OBB checkpoints with a paired detect model.",
        )
        return False

    imgsz = {
        "obb_direct": int(project.imgsz_obb_direct),
        "seq_crop_obb": int(project.imgsz_seq_crop_obb),
    }.get(role, int(project.imgsz_obb_direct))
    detect_model_path = ""
    if role == "seq_crop_obb":
        detect_model_path = (
            detectkit_latest_model_path_for_role(project, "seq_detect")
            if detectkit_latest_model_path_for_role is not None
            else ""
        )
        if not detect_model_path:
            QMessageBox.information(
                parent,
                "Quick Test",
                "The selected sequence crop OBB model needs a paired sequence-detect checkpoint before quick testing.",
            )
            return False

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
        model_path=model_path,
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
