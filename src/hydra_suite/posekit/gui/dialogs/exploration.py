#!/usr/bin/env python3
"""
Exploration dialogs: Smart Select, Embedding Explorer, Frame Metadata.
"""

import csv
import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from PySide6.QtCore import QObject, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.posekit.core.frame_grouping import group_indices_by_frame
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811

from ...core.extensions import (
    EmbeddingWorker,
    cluster_embeddings_cosine,
    pick_frames_stratified,
    select_frames_by_cluster_coverage,
)
from .utils import _load_dialog_settings, _save_dialog_settings, get_available_devices

logger = logging.getLogger("pose_label.dialogs.exploration")


class SmartSelectDialog(QDialog):
    """Dialog for smart frame selection using embeddings and clustering."""

    def __init__(
        self,
        parent,
        project,
        image_paths: List[Path],
        is_labeled_fn,
        frame_mode: bool = False,
        source_ids: Optional[List[Any]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Smart Select (Embeddings)")
        self.setMinimumSize(QSize(720, 420))
        self.project = project
        self.image_paths = image_paths
        self.is_labeled_fn = is_labeled_fn
        self.frame_mode = frame_mode
        self._frame_groups = group_indices_by_frame(
            [p.name for p in image_paths],
            source_ids if source_ids is not None else [None] * len(image_paths),
        )
        self._idx_to_frame_key = {
            i: key for key, idxs in self._frame_groups.items() for i in idxs
        }

        self._emb = None
        self._eligible_indices = None
        self._cluster = None

        layout = QVBoxLayout(self)

        if self.frame_mode:
            lbl_frame_mode = QLabel(
                "Frame Mode is ON — results grouped by source frame"
            )
            lbl_frame_mode.setStyleSheet("font-weight: bold;")
            layout.addWidget(lbl_frame_mode)

        # --- scope
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Scope"))
        self.cb_scope = QComboBox()
        self.cb_scope.addItems(["Unlabeled only", "All frames", "Labeling set"])
        self.cb_scope.setToolTip(
            "Which frames to select from:\n"
            "• Unlabeled only: Only frames not yet labeled\n"
            "• All frames: All available frames\n"
            "• Labeling set: Only frames in current labeling set"
        )
        scope_row.addWidget(self.cb_scope, 1)
        self.cb_exclude_in_labeling = QCheckBox("Exclude already in labeling set")
        self.cb_exclude_in_labeling.setChecked(True)
        self.cb_exclude_in_labeling.setToolTip(
            "Don't select frames already being labeled"
        )
        scope_row.addWidget(self.cb_exclude_in_labeling)
        layout.addLayout(scope_row)

        # --- embeddings
        emb_form = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(
            [
                # DINO v2 models (recommended for general image understanding)
                "timm/vit_base_patch14_dinov2.lvd142m",
                "timm/vit_small_patch14_dinov2.lvd142m",
                "timm/vit_large_patch14_dinov2.lvd142m",
                "timm/vit_giant_patch14_dinov2.lvd142m",
                # CLIP models
                "timm/vit_base_patch32_clip_224.openai",
                "timm/vit_bigG_14_clip_224.laion400M_e32",
                "timm/convnext_base_w_clip.laion2b_s29B_b131k_ft_in1k",
                # ResNet models
                "timm/resnet50.a1_in1k",
                "timm/resnet18.a1_in1k",
                "timm/resnet101.a1_in1k",
                # EfficientNet models
                "timm/efficientnet_b0.ra_in1k",
                "timm/efficientnet_b3.ra2_in1k",
                "timm/efficientnet_b5.sw_in12k_ft_in1k",
                # MobileNet models
                "timm/mobilenetv3_small_100.lamb_in1k",
                "timm/mobilenetv3_large_100.ra_in1k",
                # ConvNeXt models
                "timm/convnext_tiny.fb_in1k",
                "timm/convnext_base.fb_in1k",
            ]
        )
        emb_form.addRow("Embedding model", self.model_combo)

        dev_row = QHBoxLayout()
        self.dev_combo = QComboBox()
        self.dev_combo.addItems(get_available_devices())
        dev_row.addWidget(self.dev_combo)
        dev_row.addWidget(QLabel("Batch size"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(32)
        dev_row.addWidget(self.batch_spin)
        self.cb_auto_batch = QCheckBox("Auto")
        self.cb_auto_batch.setChecked(True)
        dev_row.addWidget(self.cb_auto_batch)

        dev_row.addWidget(QLabel("Max side"))
        self.max_side_spin = QSpinBox()
        self.max_side_spin.setRange(0, 2048)
        self.max_side_spin.setValue(512)
        dev_row.addWidget(self.max_side_spin)
        emb_form.addRow("Device and batch size", dev_row)

        self.cb_use_enhance = QCheckBox("Enhance images (CLAHE + unsharp)")
        self.cb_use_enhance.setChecked(bool(self.project.enhance_enabled))
        emb_form.addRow("", self.cb_use_enhance)

        layout.addLayout(emb_form)

        # --- compute button + progress
        row = QHBoxLayout()
        self.btn_compute = QPushButton("Compute / Load cached embeddings")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        row.addWidget(self.btn_compute, 1)
        row.addWidget(self.btn_cancel)
        layout.addLayout(row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%  %v / %m")
        layout.addWidget(self.progress)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        # --- selection params
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Frames to add"))
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 50000)
        self.n_spin.setValue(50)
        sel_row.addWidget(self.n_spin)

        sel_row.addWidget(QLabel("Clusters"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 5000)
        self.k_spin.setValue(20)
        sel_row.addWidget(self.k_spin)

        sel_row.addWidget(QLabel("Min per cluster"))
        self.min_per_spin = QSpinBox()
        self.min_per_spin.setRange(1, 1000)
        self.min_per_spin.setValue(1)
        sel_row.addWidget(self.min_per_spin)

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["centroid_then_diverse", "centroid"])
        sel_row.addWidget(QLabel("Strategy"))
        sel_row.addWidget(self.strategy_combo, 1)

        layout.addLayout(sel_row)

        if self.frame_mode:
            self.min_per_spin.setEnabled(False)
            self.min_per_spin.setToolTip(
                "Not used in Frame Mode — frame selection uses greedy "
                "cluster-coverage instead of per-cluster quotas."
            )
            self.strategy_combo.setEnabled(False)
            self.strategy_combo.setToolTip(
                "Not used in Frame Mode — frame selection uses greedy "
                "cluster-coverage instead of per-cluster quotas."
            )

        self.k_spin.valueChanged.connect(self._update_min_frames)
        self.min_per_spin.valueChanged.connect(self._update_min_frames)

        # --- Sampling options
        opt_row = QHBoxLayout()

        self.cb_filter_duplicates = QCheckBox("Filter near-duplicates")
        opt_row.addWidget(self.cb_filter_duplicates)

        opt_row.addWidget(QLabel("Threshold"))
        self.dup_threshold_spin = QDoubleSpinBox()
        self.dup_threshold_spin.setRange(0.5, 1.0)
        self.dup_threshold_spin.setSingleStep(0.05)
        self.dup_threshold_spin.setValue(0.95)
        self.dup_threshold_spin.setEnabled(False)
        opt_row.addWidget(self.dup_threshold_spin)
        self.cb_filter_duplicates.toggled.connect(self.dup_threshold_spin.setEnabled)

        opt_row.addWidget(QLabel("   "))
        self.cb_prefer_unlabeled = QCheckBox("Prefer unlabeled")
        self.cb_prefer_unlabeled.setChecked(True)
        opt_row.addWidget(self.cb_prefer_unlabeled)

        opt_row.addStretch()
        layout.addLayout(opt_row)

        # --- clustering method row
        cluster_row = QHBoxLayout()
        cluster_row.addWidget(QLabel("Which clustering method should be used?"))
        self.cluster_method_combo = QComboBox()
        self.cluster_method_combo.addItems(["hierarchical", "linkage"])
        self.cluster_method_combo.setCurrentIndex(0)
        cluster_row.addWidget(self.cluster_method_combo, 1)
        cluster_row.addStretch(2)
        layout.addLayout(cluster_row)

        # --- preview text
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        layout.addWidget(self.preview, 1)

        # --- bottom buttons
        # Left group: analysis actions  |  Right group: dialog actions
        bottom = QHBoxLayout()
        self.btn_preview = QPushButton("Run Clustering")
        self.btn_explorer = QPushButton("Embedding Explorer…")
        self.btn_explorer.setEnabled(False)
        self.btn_save_csv = QPushButton("Save clusters CSV…")
        self.btn_add = QPushButton("Add to Labeling Set && Close")
        self.btn_close = QPushButton("Close")
        bottom.addWidget(self.btn_preview)
        bottom.addWidget(self.btn_explorer)
        bottom.addWidget(self.btn_save_csv)
        bottom.addStretch(1)
        bottom.addWidget(self.btn_add)
        bottom.addWidget(self.btn_close)
        layout.addLayout(bottom)

        self.btn_preview.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.btn_save_csv.setEnabled(False)
        self.btn_explorer.setEnabled(False)

        # wiring
        self.btn_close.clicked.connect(self._on_close)
        self.btn_add.clicked.connect(self._on_add)
        self.btn_compute.clicked.connect(self._compute)
        self.btn_cancel.clicked.connect(self._cancel_compute)
        self.btn_preview.clicked.connect(self._preview)
        self.btn_save_csv.clicked.connect(self._save_csv)
        self.btn_explorer.clicked.connect(self._open_explorer)

        self.selected_indices: List[int] = []
        self._did_add = False

        self._thread = None
        self._worker = None

        self._update_min_frames()
        self._apply_settings()

    def _on_add(self):
        self._did_add = True
        self.accept()

    def _on_close(self):
        self.selected_indices = []
        self._did_add = False
        self.reject()

    def _apply_settings(self):
        settings = _load_dialog_settings("smart_select")
        if not settings:
            return
        self.cb_scope.setCurrentText(settings.get("scope", self.cb_scope.currentText()))
        self.cb_exclude_in_labeling.setChecked(
            bool(
                settings.get(
                    "exclude_labeling", self.cb_exclude_in_labeling.isChecked()
                )
            )
        )
        self.model_combo.setCurrentText(
            settings.get("model", self.model_combo.currentText())
        )
        self.dev_combo.setCurrentText(
            settings.get("device", self.dev_combo.currentText())
        )
        self.batch_spin.setValue(int(settings.get("batch", self.batch_spin.value())))
        self.cb_auto_batch.setChecked(
            bool(settings.get("auto_batch", self.cb_auto_batch.isChecked()))
        )
        self.max_side_spin.setValue(
            int(settings.get("max_side", self.max_side_spin.value()))
        )
        self.cb_use_enhance.setChecked(
            bool(settings.get("use_enhance", self.cb_use_enhance.isChecked()))
        )
        self.n_spin.setValue(int(settings.get("n", self.n_spin.value())))
        self.k_spin.setValue(int(settings.get("k", self.k_spin.value())))
        self.min_per_spin.setValue(
            int(settings.get("min_per", self.min_per_spin.value()))
        )
        self.strategy_combo.setCurrentText(
            settings.get("strategy", self.strategy_combo.currentText())
        )
        self.cb_filter_duplicates.setChecked(
            bool(settings.get("filter_dups", self.cb_filter_duplicates.isChecked()))
        )
        self.dup_threshold_spin.setValue(
            float(settings.get("dup_thresh", self.dup_threshold_spin.value()))
        )
        self.cb_prefer_unlabeled.setChecked(
            bool(settings.get("prefer_unlabeled", self.cb_prefer_unlabeled.isChecked()))
        )
        self.cluster_method_combo.setCurrentText(
            settings.get("cluster_method", self.cluster_method_combo.currentText())
        )

    def _save_settings(self):
        _save_dialog_settings(
            "smart_select",
            {
                "scope": self.cb_scope.currentText(),
                "exclude_labeling": bool(self.cb_exclude_in_labeling.isChecked()),
                "model": self.model_combo.currentText().strip(),
                "device": self.dev_combo.currentText(),
                "batch": int(self.batch_spin.value()),
                "auto_batch": bool(self.cb_auto_batch.isChecked()),
                "max_side": int(self.max_side_spin.value()),
                "use_enhance": bool(self.cb_use_enhance.isChecked()),
                "n": int(self.n_spin.value()),
                "k": int(self.k_spin.value()),
                "min_per": int(self.min_per_spin.value()),
                "strategy": self.strategy_combo.currentText(),
                "filter_dups": bool(self.cb_filter_duplicates.isChecked()),
                "dup_thresh": float(self.dup_threshold_spin.value()),
                "prefer_unlabeled": bool(self.cb_prefer_unlabeled.isChecked()),
                "cluster_method": self.cluster_method_combo.currentText(),
            },
        )

    def closeEvent(self, event) -> None:
        """Persist smart-select dialog settings to disk before the window closes."""
        self._save_settings()
        super().closeEvent(event)

    def _build_eligible_indices(self) -> List[int]:
        scope = self.cb_scope.currentText()
        if scope == "All frames":
            idxs = list(range(len(self.image_paths)))
        elif scope == "Labeling set":
            idxs = list(getattr(self.parent(), "labeling_frames", set()))
        else:
            # Unlabeled only
            idxs = [
                i for i, p in enumerate(self.image_paths) if not self.is_labeled_fn(p)
            ]
        if self.cb_exclude_in_labeling.isChecked():
            in_labeling = set(getattr(self.parent(), "labeling_frames", set()))
            idxs = [i for i in idxs if i not in in_labeling]
        return sorted(idxs)

    def _update_min_frames(self):
        if self.frame_mode:
            # Frame Mode doesn't use min_per_spin's per-cluster quota, so
            # the k * min_per lower bound on "frames to add" doesn't apply.
            return
        min_frames = self.k_spin.value() * self.min_per_spin.value()
        self.n_spin.setMinimum(min_frames)
        if self.n_spin.value() < min_frames:
            self.n_spin.setValue(min_frames)

    def _compute(self):
        eligible = self._build_eligible_indices()
        if not eligible:
            QMessageBox.information(
                self, "No frames", "No eligible frames in this scope."
            )
            return

        cache_dir = self.project.out_root / "posekit" / "embeddings"

        self.btn_compute.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_preview.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.btn_save_csv.setEnabled(False)
        self.preview.setPlainText("")
        self.lbl_status.setText("Starting…")
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")

        self._thread = QThread(self)
        self._worker = EmbeddingWorker(
            image_paths=self.image_paths,
            eligible_indices=eligible,
            cache_dir=cache_dir,
            model_name=self.model_combo.currentText().strip(),
            device_pref=self.dev_combo.currentText(),
            batch_size=int(self.batch_spin.value()),
            use_enhance=bool(self.cb_use_enhance.isChecked()),
            max_side=int(self.max_side_spin.value()),
            cache_ok=True,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()

    def _cancel_compute(self):
        if self._worker:
            self._worker.cancel()

    def _on_progress(self, done: int, total: int):
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(done)
        self.progress.setFormat(f"%v / {max(1, total)}  (%p%)")

    def _on_status(self, s: str):
        self.lbl_status.setText(s)

    def _on_failed(self, msg: str):
        self._cleanup_thread()
        QMessageBox.critical(self, "Embedding failed", msg)
        self.btn_compute.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def _on_finished(self, emb, eligible_indices, meta):
        self._cleanup_thread()
        self._emb = emb
        self._eligible_indices = eligible_indices
        self.lbl_status.setText(f"Embeddings ready: {emb.shape[0]} × {emb.shape[1]}")
        self.progress.setValue(self.progress.maximum())
        self.btn_compute.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_preview.setEnabled(True)
        self.btn_add.setEnabled(True)
        self.btn_save_csv.setEnabled(True)
        self.btn_explorer.setEnabled(False)

    def _cleanup_thread(self):
        if self._thread:
            self._thread.quit()
            self._thread.wait(2000)
        self._thread = None
        self._worker = None
        try:
            import gc

            gc.collect()
        except Exception:
            pass

    def _preview(self):
        if self._emb is None or self._eligible_indices is None:
            return
        n = int(self.n_spin.value())
        k = int(self.k_spin.value())
        min_per = int(self.min_per_spin.value())
        strategy = self.strategy_combo.currentText().strip()
        cluster_method = self.cluster_method_combo.currentText().strip()

        n_frames = len(self._eligible_indices)
        if cluster_method == "hierarchical" and n_frames > 2500:
            reply = QMessageBox.warning(
                self,
                "Large Dataset Warning",
                f"You have {n_frames} frames selected.\n\n"
                "Hierarchical clustering may be slow or memory-intensive for large datasets.\n\n"
                "Would you like to switch to 'linkage' method instead?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self.cluster_method_combo.setCurrentText("linkage")
                cluster_method = "linkage"
            elif reply == QMessageBox.Cancel:
                return

        cluster = cluster_embeddings_cosine(
            self._emb, k=k, method=cluster_method, seed=0
        )
        self._cluster = cluster
        self._autosave_clusters()
        self.btn_explorer.setEnabled(True)
        self.lbl_status.setText("Clusters saved. Use Export → Split method to apply.")

        if self.frame_mode:
            frame_key_of_index = {
                idx: self._idx_to_frame_key[idx]
                for idx in self._eligible_indices
                if idx in self._idx_to_frame_key
            }
            selected_keys = select_frames_by_cluster_coverage(
                cluster_id=cluster,
                eligible_indices=self._eligible_indices,
                frame_key_of_index=frame_key_of_index,
                want_n_frames=n,
            )
            self.selected_indices = sorted(
                idx for key in selected_keys for idx in self._frame_groups.get(key, [])
            )

            lines = []
            for key in selected_keys[:300]:
                frame_indices = self._frame_groups.get(key, [])
                cluster_ids_in_frame = sorted(
                    {
                        int(cluster[self._eligible_indices.index(i)])
                        for i in frame_indices
                        if i in self._eligible_indices
                    }
                )
                labeled_count = sum(
                    1
                    for i in frame_indices
                    if self.is_labeled_fn and self.is_labeled_fn(self.image_paths[i])
                )
                cluster_str = ",".join(str(c) for c in cluster_ids_in_frame)
                lines.append(
                    f"[frame {key[1]}] covers clusters {{{cluster_str}}} — "
                    f"{len(frame_indices)} instances ({labeled_count} labeled)"
                )
            if len(selected_keys) > 300:
                lines.append(f"... ({len(selected_keys) - 300} more)")
            self.preview.setPlainText("\n".join(lines))
            return

        picked = pick_frames_stratified(
            emb=self._emb,
            cluster_id=cluster,
            want_n=n,
            eligible_indices=self._eligible_indices,
            min_per_cluster=min_per,
            seed=0,
            strategy=strategy,
        )

        if self.cb_filter_duplicates.isChecked():
            try:
                # Try simple cosine similarity filtering if imported
                from ...core.extensions import filter_near_duplicates

                threshold = self.dup_threshold_spin.value()
                embeddings_picked = self._emb[
                    [self._eligible_indices.index(i) for i in picked]
                ]
                picked = filter_near_duplicates(
                    embeddings_picked,
                    list(range(len(picked))),
                    threshold=threshold,
                )
                picked = [picked[i] for i in picked]
            except Exception as e:
                logger.warning(f"Duplicate filtering failed: {e}")

        if self.cb_prefer_unlabeled.isChecked() and self.is_labeled_fn:
            labeled = [i for i in picked if self.is_labeled_fn(self.image_paths[i])]
            unlabeled = [
                i for i in picked if not self.is_labeled_fn(self.image_paths[i])
            ]
            picked = unlabeled + labeled[: max(0, n - len(unlabeled))]

        self.selected_indices = picked[:n]

        lines = []
        for idx in self.selected_indices[:300]:
            local = self._eligible_indices.index(idx)
            cid = int(cluster[local])
            labeled_str = (
                " [L]"
                if self.is_labeled_fn and self.is_labeled_fn(self.image_paths[idx])
                else ""
            )
            lines.append(f"[c{cid:03d}]{labeled_str} {self.image_paths[idx].name}")
        if len(self.selected_indices) > 300:
            lines.append(f"... ({len(self.selected_indices) - 300} more)")
        self.preview.setPlainText("\n".join(lines))

    def _open_explorer(self):
        if self._emb is None:
            QMessageBox.warning(self, "No Embeddings", "Compute embeddings first.")
            return

        dialog = EmbeddingExplorerDialog(
            embeddings=self._emb,
            image_paths=[self.image_paths[i] for i in self._eligible_indices],
            cluster_ids=self._cluster,
            is_labeled_fn=self.is_labeled_fn if self.is_labeled_fn else lambda p: False,
            parent=self,
        )

        if dialog.exec_() == QDialog.Accepted:
            explorer_selected = dialog.selected_indices
            if explorer_selected:
                global_selected = {self._eligible_indices[i] for i in explorer_selected}

                companion_count = 0
                frame_count = len(global_selected)
                if self.frame_mode:
                    keys = {
                        self._idx_to_frame_key[i]
                        for i in global_selected
                        if i in self._idx_to_frame_key
                    }
                    expanded = {
                        idx for key in keys for idx in self._frame_groups.get(key, [])
                    }
                    companions = expanded - global_selected
                    companion_count = len(companions)
                    frame_count = len(keys)
                    if companions:
                        reply = QMessageBox.question(
                            self,
                            "Added from Explorer",
                            f"This will add {frame_count} frame(s) comprising "
                            f"{len(expanded)} total instance(s), including "
                            f"{companion_count} companion instance(s), to the "
                            "labeling set. Continue?",
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.Yes,
                        )
                        if reply != QMessageBox.Yes:
                            return
                    global_selected = expanded

                existing = set(self.selected_indices)
                for idx in global_selected:
                    if idx not in existing:
                        self.selected_indices.append(idx)
                        existing.add(idx)
                QMessageBox.information(
                    self,
                    "Added from Explorer",
                    f"Added {len(global_selected)} instance(s) from "
                    f"{frame_count} frame(s).",
                )

    def _save_csv(self):
        if self._cluster is None:
            QMessageBox.information(self, "No clusters", "Run Clustering first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save cluster assignments",
            str(
                (self.project.out_root / "posekit" / "clusters").resolve()
                / "clusters.csv"
            ),
            "CSV (*.csv)",
        )
        if not path:
            return
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "cluster_id"])
            for local_pos, idx in enumerate(self._eligible_indices):
                w.writerow([str(self.image_paths[idx]), int(self._cluster[local_pos])])
        QMessageBox.information(self, "Saved", f"Wrote cluster CSV:\n{out}")

    def _autosave_clusters(self):
        if self._cluster is None:
            return
        out = self.project.out_root / "posekit" / "clusters" / "clusters.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with out.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["image", "cluster_id"])
                for local_pos, idx in enumerate(self._eligible_indices):
                    w.writerow(
                        [str(self.image_paths[idx]), int(self._cluster[local_pos])]
                    )
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight Qt-native scatter plot view (adapted from ClassKit ExplorerView)
# ──────────────────────────────────────────────────────────────────────────────


class EmbeddingExplorerView(QGraphicsView):
    """Scatter-plot widget for UMAP embeddings with hover + click signals."""

    point_hovered = Signal(int)
    point_clicked = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setStyleSheet("QGraphicsView { border: none; }")

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._coords: Optional[np.ndarray] = None
        self._cluster_ids: Optional[np.ndarray] = None
        self._labeled_mask: List[bool] = []
        self._points: List[QGraphicsEllipseItem] = []
        self._point_centers: List[tuple] = []
        self._selected_set: set = set()  # multi-selection
        self._last_hover_idx: Optional[int] = None
        self._has_fitted = False
        self._zoom_redraw_limit = 5000

    def _radius_scale(self) -> float:
        view_scale = self.transform().m11()
        if view_scale <= 0:
            view_scale = 1.0
        zoom_gain = min(0.45, max(-0.35, (view_scale - 1.0) * 0.10))
        return 1.0 - zoom_gain

    def _point_color(self, i: int) -> QColor:
        import hashlib

        if i in self._selected_set:
            return QColor(255, 245, 120)
        if self._cluster_ids is not None and i < len(self._cluster_ids):
            h = int(
                hashlib.md5(str(int(self._cluster_ids[i])).encode()).hexdigest(), 16
            )
            r = max(60, min(215, (h & 0xFF0000) >> 16))
            g = max(60, min(215, (h & 0x00FF00) >> 8))
            b = max(60, min(215, h & 0x0000FF))
            return QColor(r, g, b)
        return QColor(100, 149, 237)

    def _point_radius(self, i: int, rs: float) -> float:
        if i in self._selected_set:
            return 8.0 * rs
        if i < len(self._labeled_mask) and self._labeled_mask[i]:
            return 5.0 * rs
        return 3.5 * rs

    def _point_pen(self, i: int, rs: float) -> QPen:
        if i in self._selected_set:
            return QPen(QColor(255, 60, 60), max(2.0, 2.5 * rs))
        if i < len(self._labeled_mask) and self._labeled_mask[i]:
            return QPen(QColor(210, 210, 210), max(0.5, 0.7 * rs))
        return QPen(Qt.NoPen)

    def _point_z(self, i: int) -> float:
        if i in self._selected_set:
            return 5.0
        if i < len(self._labeled_mask) and self._labeled_mask[i]:
            return 2.0
        return 1.0

    def set_data(
        self,
        coords: np.ndarray,
        cluster_ids: Optional[np.ndarray] = None,
        labeled_mask: Optional[List[bool]] = None,
        selected_set: Optional[set] = None,
        preserve_view: bool = True,
    ):
        """Populate scatter plot from (N, 2) UMAP coords."""
        self._coords = coords
        self._cluster_ids = cluster_ids
        self._labeled_mask = labeled_mask or []
        self._selected_set = set(selected_set) if selected_set else set()

        self._scene.clear()
        self._points = []
        self._point_centers = []
        self._last_hover_idx = None

        if coords is None or len(coords) == 0:
            return

        mn = coords.min(axis=0)
        mx = coords.max(axis=0)
        span = mx - mn
        span[span == 0] = 1.0
        norm = (coords - mn) / span * 1000.0

        rs = self._radius_scale()
        for i, (x, y) in enumerate(norm):
            r = self._point_radius(i, rs)
            item = QGraphicsEllipseItem(x - r, y - r, r * 2, r * 2)
            item.setBrush(QBrush(self._point_color(i)))
            item.setPen(self._point_pen(i, rs))
            item.setZValue(self._point_z(i))
            item.setFlag(QGraphicsItem.ItemIsSelectable)
            item.setData(0, i)
            self._scene.addItem(item)
            self._points.append(item)
            self._point_centers.append((x, y))

        if not preserve_view or not self._has_fitted:
            self.fitInView(self._scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            self._has_fitted = True

    def toggle_selected(self, idx: int):
        """Fast-path: toggle a single point in/out of the selection set."""
        if idx in self._selected_set:
            self._selected_set.discard(idx)
        else:
            self._selected_set.add(idx)
        if not self._points or self._coords is None:
            return
        if idx < 0 or idx >= len(self._points):
            return
        rs = self._radius_scale()
        item = self._points[idx]
        cx, cy = self._point_centers[idx]
        r = self._point_radius(idx, rs)
        item.setRect(cx - r, cy - r, r * 2, r * 2)
        item.setBrush(QBrush(self._point_color(idx)))
        item.setPen(self._point_pen(idx, rs))
        item.setZValue(self._point_z(idx))

    def mouseMoveEvent(self, event) -> None:
        """Emit ``point_hovered`` with the index of the scatter point currently under the cursor."""
        super().mouseMoveEvent(event)
        item = self.itemAt(event.pos())
        if isinstance(item, QGraphicsEllipseItem):
            idx = item.data(0)
            if idx != self._last_hover_idx:
                self._last_hover_idx = idx
                self.point_hovered.emit(idx)
        else:
            self._last_hover_idx = None

    def mousePressEvent(self, event) -> None:
        """Emit ``point_clicked`` with the index of the scatter point clicked by the left button."""
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.pos())
            if isinstance(item, QGraphicsEllipseItem):
                self.point_clicked.emit(item.data(0))

    def wheelEvent(self, event) -> None:
        """Zoom the scatter view in or out, redrawing points when the dataset is small enough."""
        delta = event.angleDelta().y()
        if not delta:
            return
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        self.scale(factor, factor)
        if self._coords is not None and len(self._coords) <= self._zoom_redraw_limit:
            self.set_data(
                self._coords,
                cluster_ids=self._cluster_ids,
                labeled_mask=self._labeled_mask,
                selected_set=self._selected_set,
                preserve_view=True,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Embedding Explorer Dialog (Qt-native, replaces Bokeh version)
# ──────────────────────────────────────────────────────────────────────────────


class EmbeddingExplorerDialog(QDialog):
    """Interactive UMAP visualization with Qt scatter plot + image hover preview."""

    def __init__(
        self,
        embeddings: np.ndarray,
        image_paths: List[Path],
        cluster_ids: Optional[np.ndarray] = None,
        is_labeled_fn=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Embedding Explorer (UMAP)")
        self.setMinimumSize(QSize(1200, 760))

        self.embeddings: Optional[np.ndarray] = embeddings
        self.image_paths = image_paths
        self.cluster_ids = cluster_ids
        self.is_labeled_fn = is_labeled_fn or (lambda p: False)

        self.umap_projection: Optional[np.ndarray] = None
        self.selected_indices: List[int] = []
        self._umap_thread = None
        self._umap_worker = None
        self._pending_preview_idx: Optional[int] = None

        self._init_ui()

    def _init_ui(self):
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; }
            QLabel { color: #cccccc; }
            QPushButton {
                background-color: #0e639c; color: #ffffff;
                border: none; border-radius: 4px; padding: 5px 12px;
            }
            QPushButton:hover { background-color: #1177bb; }
            QPushButton:disabled { background-color: #3e3e42; color: #888888; }
            QSpinBox, QDoubleSpinBox {
                background-color: #252526; color: #e0e0e0;
                border: 1px solid #3e3e42; border-radius: 3px; padding: 3px 6px;
            }
        """)

        outer = QVBoxLayout(self)
        outer.setSpacing(6)

        # ── Controls row ─────────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Neighbors:"))
        self.neighbors_spin = QSpinBox()
        self.neighbors_spin.setRange(5, 100)
        self.neighbors_spin.setValue(15)
        ctrl.addWidget(self.neighbors_spin)

        ctrl.addWidget(QLabel("Min dist:"))
        self.min_dist_spin = QDoubleSpinBox()
        self.min_dist_spin.setRange(0.0, 1.0)
        self.min_dist_spin.setSingleStep(0.05)
        self.min_dist_spin.setValue(0.1)
        ctrl.addWidget(self.min_dist_spin)

        self.btn_compute_umap = QPushButton("Compute UMAP")
        self.btn_compute_umap.clicked.connect(self._compute_umap)
        ctrl.addWidget(self.btn_compute_umap)

        self.btn_cancel_umap = QPushButton("Cancel")
        self.btn_cancel_umap.setEnabled(False)
        self.btn_cancel_umap.clicked.connect(self._cancel_umap)
        ctrl.addWidget(self.btn_cancel_umap)

        ctrl.addStretch()
        outer.addLayout(ctrl)

        self.umap_status = QLabel("Adjust parameters and click 'Compute UMAP'.")
        self.umap_status.setStyleSheet("color: #9e9e9e; padding: 2px;")
        outer.addWidget(self.umap_status)

        # ── Main splitter: scatter (left) | preview (right) ──────────────────
        splitter = QSplitter(Qt.Horizontal)

        self.explorer_view = EmbeddingExplorerView()
        self.explorer_view.setMinimumWidth(500)
        self.explorer_view.point_hovered.connect(self._on_hover)
        self.explorer_view.point_clicked.connect(self._on_click)
        splitter.addWidget(self.explorer_view)

        # Right panel
        right = QWidget()
        right.setMinimumWidth(320)
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(6)

        right_layout.addWidget(QLabel("<b style='color:#ffffff;'>Image Preview</b>"))

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(300, 300)
        self.preview_label.setStyleSheet(
            "background-color: #252526; border: 1px solid #3e3e42; border-radius: 4px;"
        )
        self.preview_label.setText(
            "<span style='color:#555555;'>Hover a point to preview</span>"
        )
        right_layout.addWidget(self.preview_label, 1)

        self.preview_info = QLabel()
        self.preview_info.setWordWrap(True)
        self.preview_info.setTextFormat(Qt.RichText)
        self.preview_info.setStyleSheet("color: #cccccc; padding: 4px;")
        self.preview_info.setText(
            "<span style='color:#9e9e9e;'>Hover to preview  ·  Click to select / deselect</span>"
        )
        right_layout.addWidget(self.preview_info)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3e3e42;")
        right_layout.addWidget(sep)

        self.selection_summary = QLabel("No points selected.")
        self.selection_summary.setStyleSheet(
            "color: #4ec9b0; padding: 4px; font-weight: bold;"
        )
        right_layout.addWidget(self.selection_summary)

        btn_row = QHBoxLayout()
        self.btn_clear_sel = QPushButton("Clear Selection")
        self.btn_clear_sel.clicked.connect(self._clear_selection)
        self.btn_add_sel = QPushButton("Add to Labeling Set")
        self.btn_add_sel.clicked.connect(self.accept)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_clear_sel)
        btn_row.addWidget(self.btn_add_sel)
        btn_row.addWidget(btn_close)
        right_layout.addLayout(btn_row)

        splitter.addWidget(right)
        splitter.setSizes([720, 380])
        outer.addWidget(splitter, 1)

        # Debounce timer for preview
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(60)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._flush_preview)

    # ── UMAP ─────────────────────────────────────────────────────────────────

    def _compute_umap(self):
        if self.embeddings is None:
            return
        self.btn_compute_umap.setEnabled(False)
        self.btn_cancel_umap.setEnabled(True)
        self.umap_status.setText("Computing UMAP projection...")
        self._umap_thread = QThread(self)
        self._umap_worker = UMAPWorker(
            embeddings=self.embeddings,
            n_neighbors=self.neighbors_spin.value(),
            min_dist=self.min_dist_spin.value(),
            random_state=42,
        )
        self._umap_worker.moveToThread(self._umap_thread)
        self._umap_thread.started.connect(self._umap_worker.run)
        self._umap_worker.progress.connect(self.umap_status.setText)
        self._umap_worker.finished.connect(self._on_umap_finished)
        self._umap_worker.failed.connect(self._on_umap_failed)
        self._umap_thread.start()

    def _cancel_umap(self):
        if self._umap_worker:
            self._umap_worker.cancel()

    def _on_umap_finished(self, projection: np.ndarray):
        if self._umap_thread:
            self._umap_thread.quit()
            self._umap_thread.wait(2000)
        self._umap_thread = None
        self._umap_worker = None
        self.umap_projection = projection
        n = projection.shape[0]
        self.umap_status.setText(
            f"UMAP complete  ·  {n:,} points  (hover to preview, click to select)"
        )
        self.btn_compute_umap.setEnabled(True)
        self.btn_cancel_umap.setEnabled(False)
        labeled_mask = [self.is_labeled_fn(p) for p in self.image_paths]
        self.explorer_view.set_data(
            projection,
            cluster_ids=self.cluster_ids,
            labeled_mask=labeled_mask,
            selected_set=set(self.selected_indices),
        )

    def _on_umap_failed(self, msg: str):
        if self._umap_thread:
            self._umap_thread.quit()
            self._umap_thread.wait(2000)
        self._umap_thread = None
        self._umap_worker = None
        self.umap_status.setText(f"UMAP failed: {msg}")
        self.btn_compute_umap.setEnabled(True)
        self.btn_cancel_umap.setEnabled(False)
        QMessageBox.critical(self, "UMAP Error", msg)

    # ── Preview ───────────────────────────────────────────────────────────────

    def _on_hover(self, idx: int):
        self._pending_preview_idx = idx
        if not self._preview_timer.isActive():
            self._preview_timer.start()

    def _flush_preview(self):
        idx = self._pending_preview_idx
        if idx is not None:
            self._load_preview(idx)

    def _load_preview(self, idx: int):
        if idx < 0 or idx >= len(self.image_paths):
            return
        path = self.image_paths[idx]

        try:
            pixmap = QPixmap(str(path))

            if not pixmap.isNull():
                preview_w = max(8, self.preview_label.width() - 8)
                preview_h = max(8, self.preview_label.height() - 8)
                pixmap = pixmap.scaled(
                    preview_w,
                    preview_h,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self.preview_label.setPixmap(pixmap)
            else:
                self.preview_label.setText(
                    "<span style='color:#f44336;'>Could not load image</span>"
                )
        except Exception as exc:
            self.preview_label.setText(
                f"<span style='color:#f44336;'>Error: {exc}</span>"
            )

        cluster_str = "n/a"
        if self.cluster_ids is not None and idx < len(self.cluster_ids):
            cluster_str = str(int(self.cluster_ids[idx]))
        labeled_str = "yes" if self.is_labeled_fn(path) else "no"
        selected_str = "yes" if idx in self.selected_indices else "no"
        self.preview_info.setText(
            f"<div style='line-height:1.6;'>"
            f"<b>Index:</b> {idx} &nbsp; <b>Cluster:</b> {cluster_str}<br>"
            f"<b>Labeled:</b> {labeled_str} &nbsp; <b>Selected:</b> {selected_str}<br>"
            f"<span style='color:#9e9e9e; font-size:11px;'>{path.name}</span>"
            f"</div>"
        )

    # ── Selection ─────────────────────────────────────────────────────────────

    def _on_click(self, idx: int):
        if idx in self.selected_indices:
            self.selected_indices.remove(idx)
        else:
            self.selected_indices.append(idx)
        self._update_selection_summary()
        self.explorer_view.toggle_selected(idx)
        self._load_preview(idx)

    def _update_selection_summary(self):
        n = len(self.selected_indices)
        if n == 0:
            self.selection_summary.setText("No points selected.")
        else:
            self.selection_summary.setText(
                f"{n:,} point(s) selected  \u2014  click \u2018Add to Labeling Set\u2019 to use them."
            )

    def _clear_selection(self):
        self.selected_indices.clear()
        self._update_selection_summary()
        if self.umap_projection is not None:
            labeled_mask = [self.is_labeled_fn(p) for p in self.image_paths]
            self.explorer_view.set_data(
                self.umap_projection,
                cluster_ids=self.cluster_ids,
                labeled_mask=labeled_mask,
                selected_set=set(),
            )

    def closeEvent(self, event) -> None:
        """Cancel any running UMAP worker, release embedding memory, and close the dialog."""
        if self._umap_worker:
            self._umap_worker.cancel()
        if self._umap_thread:
            self._umap_thread.quit()
            self._umap_thread.wait(1000)
        self.embeddings = None
        self._canon_cache.clear()
        super().closeEvent(event)


class UMAPWorker(QObject):
    """Background worker for UMAP projection."""

    progress = Signal(str)
    finished = Signal(np.ndarray)
    failed = Signal(str)

    def __init__(
        self,
        embeddings: np.ndarray,
        n_neighbors: int,
        min_dist: float = 0.1,
        random_state: int = 42,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self._cancelled = False

    def cancel(self):
        """Request early termination of the UMAP projection computation."""
        self._cancelled = True

    def run(self):
        """Fit a 2-D UMAP projection on the stored embeddings and emit the result array."""
        try:
            if self._cancelled:
                return
            self.progress.emit("Importing UMAP...")
            import umap

            if self._cancelled:
                return
            self.progress.emit(
                f"Running UMAP  ({self.n_neighbors} neighbors, min_dist={self.min_dist})..."
            )
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric="cosine",
                random_state=self.random_state,
            )
            projection = reducer.fit_transform(self.embeddings)
            if self._cancelled:
                return
            self.finished.emit(projection.astype(np.float32))
        except Exception as exc:
            self.failed.emit(str(exc))
            self.failed.emit(str(exc))
            self.failed.emit(str(exc))
