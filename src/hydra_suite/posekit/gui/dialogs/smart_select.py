"""Smart Select: pick the next frames to label, the way ClassKit picks candidates.

Embeddings of every frame are clustered (K-means, as in ClassKit) and a batch
is built from a mix of signals — frames that look like known model errors,
frames the model is unsure about, rare clusters, gaps the labeled set leaves,
typical under-labeled frames and random checks (see
:mod:`hydra_suite.posekit.core.selection`).  The result is an ordered candidate
list; *Start Labeling* walks it in that order.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PySide6.QtCore import QSize, Qt, QThread
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.core.selection.clustering import auto_cluster_count
from hydra_suite.posekit.core.frame_grouping import group_indices_by_frame
from hydra_suite.posekit.core.selection import (
    REASON_AUDIT,
    REASON_DIVERSITY,
    REASON_ERROR,
    REASON_LABELS,
    REASON_RARE,
    REASON_REPRESENTATIVE,
    REASON_UNCERTAINTY,
    REASONS,
    PoseBatchConfig,
)
from hydra_suite.posekit.core.smart_select import (
    SmartSelectOutcome,
    SmartSelectRequest,
)
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog

from ...core.extensions import EmbeddingWorker
from ..workers import SmartSelectWorker
from .utils import _load_dialog_settings, _save_dialog_settings, get_available_devices

logger = logging.getLogger("pose_label.dialogs.smart_select")

REASON_EXPLORER = "explorer"

# Threads still running when their dialog closed: kept referenced (and
# unparented) until they finish, so Qt never destroys a running thread.
_ORPHANED_THREADS: List[QThread] = []


def _orphan_until_finished(thread: QThread) -> None:
    if thread is None or thread.isFinished():
        return
    thread.setParent(None)
    _ORPHANED_THREADS.append(thread)

    def _release() -> None:
        if thread in _ORPHANED_THREADS:
            _ORPHANED_THREADS.remove(thread)

    thread.finished.connect(_release)


_REASON_TEXT = dict(REASON_LABELS, **{REASON_EXPLORER: "Picked in Explorer"})

EMBEDDING_MODELS = [
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

CLUSTER_METHODS = [
    ("K-means", "kmeans"),
    ("MiniBatch K-means", "minibatch"),
    ("HDBSCAN (slow above ~10,000 frames)", "hdbscan"),
]

SCOPES = ["Unlabeled only", "All frames", "Labeling set"]

# (attribute, label, default %, tooltip)
_MIX_FIELDS = [
    (
        "error_spin",
        "Near model errors",
        20,
        "Unlabeled frames that look like labeled frames the model gets wrong "
        "(pose error above the threshold). Needs model predictions.",
    ),
    (
        "uncertainty_spin",
        "Model unsure",
        25,
        "Frames where the model's keypoint confidence is lowest. Needs model "
        "predictions.",
    ),
    (
        "rare_spin",
        "Rare clusters",
        10,
        "Frames from small or barely-labeled clusters, sampled with weight "
        "size^-alpha.",
    ),
    (
        "diversity_spin",
        "Fill gaps",
        30,
        "Frames in the biggest uncovered regions: each pick most reduces the "
        "distance from unlabeled frames to their nearest labeled one, so the "
        "labeled set covers the whole dataset.",
    ),
    (
        "representative_spin",
        "Typical",
        10,
        "Frames from dense clusters that have few labels.",
    ),
]


@dataclass
class PredictionSource:
    """Cached pose predictions of the main window's current model."""

    label: str  # e.g. "best.pt (yolo)"
    predictions: Dict[str, list]


class SmartSelectDialog(QDialog):
    """Build an ordered batch of frames to label next."""

    def __init__(
        self,
        parent,
        project,
        image_paths: List[Path],
        is_labeled_fn: Callable[[Path], bool],
        frame_mode: bool = False,
        source_ids: Optional[List[Any]] = None,
        labeling_frames: Optional[set] = None,
        label_path_fn: Optional[Callable[[Path], Path]] = None,
        prediction_source: Optional[PredictionSource] = None,
        no_prediction_reason: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Smart Select")
        self.setMinimumSize(QSize(980, 720))
        self.project = project
        self.image_paths = list(image_paths)
        self.is_labeled_fn = is_labeled_fn or (lambda p: False)
        self.frame_mode = bool(frame_mode)
        self.labeling_frames = set(
            labeling_frames
            if labeling_frames is not None
            else getattr(parent, "labeling_frames", set())
        )
        self.label_path_fn = label_path_fn
        self.prediction_source = prediction_source
        self.no_prediction_reason = no_prediction_reason
        self._source_ids = (
            list(source_ids) if source_ids is not None else [None] * len(image_paths)
        )
        self._frame_groups = group_indices_by_frame(
            [p.name for p in self.image_paths], self._source_ids
        )
        self._idx_to_frame_key = {
            i: key for key, idxs in self._frame_groups.items() for i in idxs
        }

        self._emb_all: Optional[np.ndarray] = None
        self._emb_key: Optional[tuple] = None
        self._eligible_indices: List[int] = []
        self._outcome: Optional[SmartSelectOutcome] = None
        self.batch_units: List[int] = []
        self._unit_reason: Dict[int, str] = {}
        self._emb_thread: Optional[QThread] = None
        self._emb_worker: Optional[EmbeddingWorker] = None
        self._sel_worker: Optional[SmartSelectWorker] = None
        self._busy = False
        self._cancel_requested = False

        # Results read by the caller after accept().
        self.selected_indices: List[int] = []
        self.start_labeling = False
        self._did_add = False

        self._build_ui()
        self._apply_settings()
        self._on_auto_k_toggled(self.cb_auto_k.isChecked())
        self._update_model_label()
        self._update_remainder()
        self._set_busy(False)

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        if self.frame_mode:
            banner = QLabel(
                "Frame Mode is on — Smart Select picks whole source frames; "
                "every individual on a picked frame is added."
            )
            banner.setWordWrap(True)
            banner.setStyleSheet("font-weight: bold;")
            layout.addWidget(banner)

        top = QHBoxLayout()
        layout.addLayout(top)

        # --- Batch settings
        batch_box = QGroupBox("Batch")
        batch_form = QFormLayout(batch_box)

        scope_row = QHBoxLayout()
        self.cb_scope = QComboBox()
        self.cb_scope.addItems(SCOPES)
        self.cb_scope.setToolTip(
            "Which frames may be picked:\n"
            "• Unlabeled only: frames without a saved label\n"
            "• All frames: any frame (labeled ones are re-checked)\n"
            "• Labeling set: reorder the frames already in the labeling set"
        )
        scope_row.addWidget(self.cb_scope, 1)
        self.cb_exclude_in_labeling = QCheckBox("Skip frames in labeling set")
        self.cb_exclude_in_labeling.setChecked(True)
        self.cb_exclude_in_labeling.setToolTip(
            "Don't pick frames that are already in the labeling set."
        )
        scope_row.addWidget(self.cb_exclude_in_labeling)
        batch_form.addRow("Pick from", scope_row)

        size_row = QHBoxLayout()
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 50000)
        self.n_spin.setValue(50)
        self.n_spin.setToolTip(
            "Number of frames in the batch"
            + (" (source frames in Frame Mode)." if self.frame_mode else ".")
        )
        size_row.addWidget(self.n_spin)
        size_row.addSpacing(12)
        size_row.addWidget(QLabel("Clusters"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(2, 5000)
        self.k_spin.setValue(50)
        size_row.addWidget(self.k_spin)
        self.cb_auto_k = QCheckBox("Auto")
        self.cb_auto_k.setChecked(True)
        self.cb_auto_k.setToolTip(
            "Use about √N clusters (N = frames), like ClassKit's over-clustering."
        )
        self.cb_auto_k.toggled.connect(self._on_auto_k_toggled)
        size_row.addWidget(self.cb_auto_k)
        size_row.addSpacing(12)
        self.cluster_method_combo = QComboBox()
        for text, value in CLUSTER_METHODS:
            self.cluster_method_combo.addItem(text, value)
        self.cluster_method_combo.setToolTip(
            "K-means is deterministic and recommended (MiniBatch is used "
            "automatically above 5,000 frames)."
        )
        size_row.addWidget(self.cluster_method_combo)
        size_row.addStretch(1)
        batch_form.addRow("Frames", size_row)
        top.addWidget(batch_box, 3)

        # --- Mix
        mix_box = QGroupBox("Mix")
        mix = QGridLayout(mix_box)
        for row, (attr, label, default, tip) in enumerate(_MIX_FIELDS):
            lbl = QLabel(label)
            lbl.setToolTip(tip)
            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setSuffix(" %")
            spin.setValue(default)
            spin.setToolTip(tip)
            spin.valueChanged.connect(self._update_remainder)
            setattr(self, attr, spin)
            mix.addWidget(lbl, row, 0)
            mix.addWidget(spin, row, 1)
        self.remainder_label = QLabel()
        self.remainder_label.setToolTip(
            "Whatever the mix leaves goes to random spot checks."
        )
        mix.addWidget(self.remainder_label, len(_MIX_FIELDS), 0, 1, 2)

        mix.addWidget(QLabel("Rarity α"), 0, 2)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(0.75)
        self.alpha_spin.setToolTip(
            "0 = clusters in proportion to their size, 1 = every cluster equal."
        )
        mix.addWidget(self.alpha_spin, 0, 3)
        mix.addWidget(QLabel("Error above"), 1, 2)
        self.error_thr_spin = QDoubleSpinBox()
        self.error_thr_spin.setRange(0.0, 1.0)
        self.error_thr_spin.setDecimals(3)
        self.error_thr_spin.setSingleStep(0.01)
        self.error_thr_spin.setValue(0.05)
        self.error_thr_spin.setToolTip(
            "A labeled frame counts as a model error when its mean keypoint "
            "error, divided by the object size, is above this (0.05 = the "
            "Evaluation dialog's PCK threshold)."
        )
        mix.addWidget(self.error_thr_spin, 1, 3)
        self.cb_dedup = QCheckBox("Skip near-duplicates")
        self.cb_dedup.setToolTip(
            "Never put two frames this similar (cosine) in the same batch."
        )
        mix.addWidget(self.cb_dedup, 2, 2)
        self.dedup_spin = QDoubleSpinBox()
        self.dedup_spin.setRange(0.5, 1.0)
        self.dedup_spin.setSingleStep(0.01)
        self.dedup_spin.setValue(0.97)
        self.dedup_spin.setEnabled(False)
        self.cb_dedup.toggled.connect(self.dedup_spin.setEnabled)
        mix.addWidget(self.dedup_spin, 2, 3)
        top.addWidget(mix_box, 2)

        self.model_label = QLabel()
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_label)

        # --- Embeddings (collapsed details)
        emb_box = QGroupBox("Embeddings")
        emb_form = QFormLayout(emb_box)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(EMBEDDING_MODELS)
        emb_form.addRow("Model", self.model_combo)
        dev_row = QHBoxLayout()
        self.dev_combo = QComboBox()
        self.dev_combo.addItems(get_available_devices())
        dev_row.addWidget(self.dev_combo)
        dev_row.addWidget(QLabel("Batch size"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(32)
        dev_row.addWidget(self.batch_spin)
        dev_row.addWidget(QLabel("Max side"))
        self.max_side_spin = QSpinBox()
        self.max_side_spin.setRange(0, 2048)
        self.max_side_spin.setValue(512)
        self.max_side_spin.setToolTip("Downscale images to this size first (0 = off).")
        dev_row.addWidget(self.max_side_spin)
        self.cb_use_enhance = QCheckBox("Enhance (CLAHE + unsharp)")
        self.cb_use_enhance.setChecked(
            bool(getattr(self.project, "enhance_enabled", False))
        )
        dev_row.addWidget(self.cb_use_enhance)
        dev_row.addStretch(1)
        emb_form.addRow("Device", dev_row)
        layout.addWidget(emb_box)

        # --- Build row
        build_row = QHBoxLayout()
        self.btn_build = QPushButton("Build Batch")
        self.btn_build.setDefault(True)
        self.btn_build.setMinimumHeight(32)
        self.btn_build.setToolTip(
            "Compute (or load cached) embeddings, cluster every frame and pick "
            "the batch."
        )
        self.btn_cancel = QPushButton("Cancel")
        build_row.addWidget(self.btn_build, 1)
        build_row.addWidget(self.btn_cancel)
        layout.addLayout(build_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        # --- Candidates + preview
        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        self.summary_label = QLabel("No batch yet.")
        self.summary_label.setWordWrap(True)
        left_l.addWidget(self.summary_label)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "Frame", "Cluster", "Why", "Detail"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        left_l.addWidget(self.table, 1)
        tbl_btns = QHBoxLayout()
        self.btn_remove = QPushButton("Remove from batch")
        self.btn_remove.setToolTip("Drop the selected candidates (Delete).")
        self.btn_remove.clicked.connect(self._remove_selected_rows)
        self.btn_explorer = QPushButton("Embedding Explorer…")
        self.btn_explorer.setToolTip(
            "See every frame on a UMAP map with the batch highlighted; click "
            "points to add or drop them."
        )
        self.btn_save_csv = QPushButton("Save clusters CSV…")
        tbl_btns.addWidget(self.btn_remove)
        tbl_btns.addWidget(self.btn_explorer)
        tbl_btns.addWidget(self.btn_save_csv)
        tbl_btns.addStretch(1)
        left_l.addLayout(tbl_btns)
        split.addWidget(left)

        self.preview = QLabel("Select a candidate to preview it.")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(240, 240)
        self.preview.setWordWrap(True)
        self.preview.setStyleSheet("border: 1px solid #3e3e42; border-radius: 4px;")
        split.addWidget(self.preview)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)

        # --- Dialog buttons
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.btn_close = QPushButton("Close")
        self.btn_add = QPushButton("Add to Labeling Set")
        self.btn_add.setToolTip("Add the batch to the labeling set and close.")
        self.btn_start = QPushButton("Start Labeling")
        self.btn_start.setToolTip(
            "Add the batch to the labeling set, sort the list in batch order "
            "and open the first frame."
        )
        self.btn_start.setMinimumHeight(32)
        self.btn_start.setStyleSheet(
            "QPushButton { background:#2e7d32; color:white; font-weight:bold; "
            "padding: 4px 16px; } QPushButton:disabled { background:#3a3a3a; "
            "color:#777; }"
        )
        bottom.addWidget(self.btn_close)
        bottom.addWidget(self.btn_add)
        bottom.addWidget(self.btn_start)
        layout.addLayout(bottom)

        self.btn_build.clicked.connect(self._build)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_explorer.clicked.connect(self._open_explorer)
        self.btn_save_csv.clicked.connect(self._save_csv)
        self.btn_close.clicked.connect(self.reject)
        self.btn_add.clicked.connect(lambda: self._finish(start=False))
        self.btn_start.clicked.connect(lambda: self._finish(start=True))
        for key in (QKeySequence.Delete, QKeySequence(Qt.Key_Backspace)):
            sc = QShortcut(key, self.table)
            sc.setContext(Qt.WidgetShortcut)
            sc.activated.connect(self._remove_selected_rows)

    def _unit_count(self) -> int:
        return len(self._frame_groups) if self.frame_mode else len(self.image_paths)

    def _on_auto_k_toggled(self, on: bool) -> None:
        """Auto shows the cluster count it will use (≈ √N, clamped)."""
        self.k_spin.setEnabled(not on)
        if on:
            self.k_spin.setValue(
                max(self.k_spin.minimum(), auto_cluster_count(self._unit_count()))
            )

    def _mix_total(self) -> int:
        return sum(getattr(self, attr).value() for attr, *_ in _MIX_FIELDS)

    def _update_remainder(self) -> None:
        total = self._mix_total()
        if total > 100:
            self.remainder_label.setText(
                f"Mix adds up to {total}% — scaled down to fit; no random checks."
            )
        else:
            self.remainder_label.setText(f"Random checks: {100 - total}%")

    def _update_model_label(self) -> None:
        src = self.prediction_source
        has_preds = bool(src and src.predictions)
        for attr in ("error_spin", "uncertainty_spin"):
            getattr(self, attr).setEnabled(has_preds)
        self.error_thr_spin.setEnabled(has_preds)
        if not has_preds:
            why = self.no_prediction_reason or "No cached pose predictions."
            self.model_label.setText(
                f"<b>Model signals off.</b> {why} Run <i>Predict Dataset…</i> "
                "with a pose model to add error and uncertainty picks; their "
                "share goes to <i>Fill gaps</i> until then."
            )
            return
        n_cached = sum(
            1
            for p in self.image_paths
            if str(p) in src.predictions or str(Path(p).resolve()) in src.predictions
        )
        self.model_label.setText(
            f"<b>Model signals:</b> {src.label} — predictions cached for "
            f"{n_cached:,} of {len(self.image_paths):,} frames."
            + (
                ""
                if n_cached == len(self.image_paths)
                else " Frames without a prediction can still be picked by "
                "the other slots."
            )
        )

    def _set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self.btn_build.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        has_batch = bool(self.batch_units) and not busy
        self.btn_add.setEnabled(has_batch)
        self.btn_start.setEnabled(has_batch)
        self.btn_remove.setEnabled(has_batch)
        self.btn_explorer.setEnabled(self._outcome is not None and not busy)
        self.btn_save_csv.setEnabled(self._outcome is not None and not busy)

    # ── settings ──────────────────────────────────────────────────────────

    def _apply_settings(self) -> None:
        s = _load_dialog_settings("smart_select") or {}
        if not isinstance(s, dict) or not s:
            return

        def num(key, spin, cast=int, src=None):
            """Restore a spin box; ignore missing or malformed values."""
            src = s if src is None else src
            try:
                spin.setValue(cast(src[key]))
            except (KeyError, TypeError, ValueError, OverflowError):
                pass

        def flag(key, box, default):
            value = s.get(key, default)
            if isinstance(value, bool):
                box.setChecked(value)

        def text(key, combo):
            value = s.get(key)
            if isinstance(value, str) and value.strip():
                combo.setCurrentText(value)

        if s.get("scope") in SCOPES:
            self.cb_scope.setCurrentText(s["scope"])
        flag("exclude_labeling", self.cb_exclude_in_labeling, True)
        text("model", self.model_combo)
        if s.get("device") in [
            self.dev_combo.itemText(i) for i in range(self.dev_combo.count())
        ]:
            self.dev_combo.setCurrentText(s["device"])
        num("batch", self.batch_spin)
        num("max_side", self.max_side_spin)
        flag("use_enhance", self.cb_use_enhance, self.cb_use_enhance.isChecked())
        num("n", self.n_spin)
        num("k", self.k_spin)
        flag("auto_k", self.cb_auto_k, True)
        i = self.cluster_method_combo.findData(s.get("cluster_method_v2"))
        if i >= 0:
            self.cluster_method_combo.setCurrentIndex(i)
        mix = s.get("mix")
        if isinstance(mix, dict):
            for attr, *_ in _MIX_FIELDS:
                num(attr, getattr(self, attr), src=mix)
        num("rare_alpha", self.alpha_spin, float)
        num("error_threshold", self.error_thr_spin, float)
        flag("dedup", self.cb_dedup, False)
        num("dedup_threshold", self.dedup_spin, float)
        self.k_spin.setEnabled(not self.cb_auto_k.isChecked())
        self.dedup_spin.setEnabled(self.cb_dedup.isChecked())

    def _save_settings(self) -> None:
        _save_dialog_settings(
            "smart_select",
            {
                "scope": self.cb_scope.currentText(),
                "exclude_labeling": bool(self.cb_exclude_in_labeling.isChecked()),
                "model": self.model_combo.currentText().strip(),
                "device": self.dev_combo.currentText(),
                "batch": int(self.batch_spin.value()),
                "max_side": int(self.max_side_spin.value()),
                "use_enhance": bool(self.cb_use_enhance.isChecked()),
                "n": int(self.n_spin.value()),
                "k": int(self.k_spin.value()),
                "auto_k": bool(self.cb_auto_k.isChecked()),
                "cluster_method_v2": self.cluster_method_combo.currentData(),
                "mix": {
                    attr: int(getattr(self, attr).value()) for attr, *_ in _MIX_FIELDS
                },
                "rare_alpha": float(self.alpha_spin.value()),
                "error_threshold": float(self.error_thr_spin.value()),
                "dedup": bool(self.cb_dedup.isChecked()),
                "dedup_threshold": float(self.dedup_spin.value()),
            },
        )

    def done(self, result: int) -> None:  # accept, reject and Esc all land here
        """Stop background work and remember the settings, however it closes."""
        self._stop_workers()
        try:
            self._save_settings()
        except Exception:  # settings are a convenience; never block closing
            logger.debug("Could not save Smart Select settings", exc_info=True)
        super().done(result)

    def closeEvent(self, event) -> None:
        """Route the title-bar close through ``done`` (stops workers, saves)."""
        event.ignore()
        self.reject()

    # ── eligibility ───────────────────────────────────────────────────────

    def _build_eligible_indices(self) -> List[int]:
        scope = self.cb_scope.currentText()
        if scope == "All frames":
            idxs = list(range(len(self.image_paths)))
        elif scope == "Labeling set":
            idxs = sorted(
                i for i in self.labeling_frames if 0 <= i < len(self.image_paths)
            )
        else:
            idxs = [
                i for i, p in enumerate(self.image_paths) if not self.is_labeled_fn(p)
            ]
        if self.cb_exclude_in_labeling.isChecked() and scope != "Labeling set":
            idxs = [i for i in idxs if i not in self.labeling_frames]
        return idxs

    def _embedding_key(self) -> tuple:
        return (
            self.model_combo.currentText().strip(),
            bool(self.cb_use_enhance.isChecked()),
            int(self.max_side_spin.value()),
        )

    # ── build: embeddings, then the selection job ─────────────────────────

    def _build(self) -> None:
        if self._busy or not self.image_paths:
            return
        self._cancel_requested = False
        eligible = self._build_eligible_indices()
        if not eligible:
            QMessageBox.information(
                self, "Nothing to pick", "No frames match the 'Pick from' choice."
            )
            return
        self._eligible_indices = eligible
        if self._emb_all is not None and self._emb_key == self._embedding_key():
            self._start_selection()
            return
        self._compute_embeddings()

    def _compute_embeddings(self) -> None:
        self._set_busy(True)
        self.lbl_status.setText("Loading embeddings…")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._emb_thread = QThread(self)
        self._emb_worker = EmbeddingWorker(
            image_paths=self.image_paths,
            eligible_indices=self._eligible_indices,
            cache_dir=Path(self.project.out_root) / "posekit",
            model_name=self.model_combo.currentText().strip(),
            device_pref=self.dev_combo.currentText(),
            batch_size=int(self.batch_spin.value()),
            use_enhance=bool(self.cb_use_enhance.isChecked()),
            max_side=int(self.max_side_spin.value()),
            cache_ok=True,
        )
        self._emb_key_pending = self._embedding_key()
        self._emb_worker.moveToThread(self._emb_thread)
        self._emb_thread.started.connect(self._emb_worker.run)
        self._emb_worker.progress.connect(self._on_emb_progress)
        self._emb_worker.status.connect(self.lbl_status.setText)
        self._emb_worker.failed.connect(self._on_emb_failed)
        self._emb_worker.finished.connect(self._on_emb_finished)
        self._emb_thread.start()

    def _on_emb_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.progress.setFormat(f"Embedding %v / {max(1, total)}")

    def _end_emb_thread(self) -> None:
        if self._emb_thread is not None:
            self._emb_thread.quit()
            self._emb_thread.wait(3000)
        self._emb_thread = None
        self._emb_worker = None

    def _is_current(self, worker) -> bool:
        """Signals from a worker this dialog has since replaced are ignored."""
        sender = self.sender()
        return sender is None or sender is worker

    def _on_emb_failed(self, msg: str) -> None:
        if not self._is_current(self._emb_worker):
            return
        self._end_emb_thread()
        self._set_busy(False)
        self.progress.setFormat("%p%")
        self.progress.setValue(0)
        if msg == "Canceled.":
            self.lbl_status.setText("Canceled.")
            return
        self.lbl_status.setText("Embedding failed.")
        QMessageBox.critical(self, "Embedding failed", msg)

    def _on_emb_finished(self, emb_all, _eligible, _meta) -> None:
        if not self._is_current(self._emb_worker):
            return
        self._end_emb_thread()
        self._remove_legacy_embedding_cache()
        emb_all = np.asarray(emb_all)
        if emb_all.ndim != 2 or len(emb_all) != len(self.image_paths):
            self._set_busy(False)
            QMessageBox.critical(
                self, "Embedding failed", "Embeddings do not match the frame list."
            )
            return
        self._emb_all = emb_all
        self._emb_key = self._emb_key_pending
        self.progress.setFormat("%p%")
        if self._cancel_requested:
            # Canceled after the last check inside the worker: keep the
            # embeddings (the next Build reuses them) but stop here.
            self.progress.setValue(0)
            self.lbl_status.setText("Canceled.")
            self._set_busy(False)
            return
        self._start_selection()

    def _remove_legacy_embedding_cache(self) -> None:
        """Drop caches written by the previous Smart Select.

        They sat one folder deeper (``posekit/embeddings/embeddings``) under
        a key that included device and batch size, so nothing reads or
        cleans them any more.
        """
        legacy = Path(self.project.out_root) / "posekit" / "embeddings" / "embeddings"
        if not legacy.is_dir():
            return
        import shutil

        try:
            shutil.rmtree(legacy)
            logger.info("Removed the old Smart Select embedding cache: %s", legacy)
        except OSError:
            logger.debug("Could not remove %s", legacy, exc_info=True)

    def _start_selection(self) -> None:
        n = len(self.image_paths)
        labeled = np.array([bool(self.is_labeled_fn(p)) for p in self.image_paths])
        cand = np.zeros(n, dtype=bool)
        cand[self._eligible_indices] = True
        preds = self.prediction_source.predictions if self.prediction_source else None
        label_paths: Dict[int, Path] = {}
        if preds and self.label_path_fn is not None:
            for i in np.where(labeled)[0]:
                try:
                    label_paths[int(i)] = Path(self.label_path_fn(self.image_paths[i]))
                except Exception:
                    continue
        # The model slots keep their share even without predictions: with no
        # signal they pick nothing and the core hands their budget to the
        # "Fill gaps" top-up (zeroing them here would turn it into random
        # checks instead).
        cfg = PoseBatchConfig(
            batch_size=int(self.n_spin.value()),
            error_fraction=self.error_spin.value() / 100.0,
            uncertainty_fraction=self.uncertainty_spin.value() / 100.0,
            rare_fraction=self.rare_spin.value() / 100.0,
            rare_alpha=float(self.alpha_spin.value()),
            diversity_fraction=self.diversity_spin.value() / 100.0,
            representative_fraction=self.representative_spin.value() / 100.0,
            error_threshold=float(self.error_thr_spin.value()),
            dedup_threshold=(
                float(self.dedup_spin.value()) if self.cb_dedup.isChecked() else None
            ),
        )
        unit_keys = (
            [self._idx_to_frame_key.get(i, ("_idx", i)) for i in range(n)]
            if self.frame_mode
            else None
        )
        request = SmartSelectRequest(
            embeddings=self._emb_all,
            image_paths=self.image_paths,
            candidate_mask=cand,
            labeled_mask=labeled,
            config=cfg,
            unit_keys=unit_keys,
            n_clusters=0 if self.cb_auto_k.isChecked() else int(self.k_spin.value()),
            cluster_method=self.cluster_method_combo.currentData() or "kmeans",
            predictions=preds,
            label_paths=label_paths,
            num_keypoints=len(getattr(self.project, "keypoint_names", []) or []),
        )
        self._set_busy(True)
        self.progress.setRange(0, 0)  # busy
        self._sel_worker = SmartSelectWorker(request, self)
        self._sel_worker.status.connect(self.lbl_status.setText)
        self._sel_worker.result.connect(self._on_selection_ready)
        self._sel_worker.error.connect(self._on_selection_failed)
        self._sel_worker.canceled.connect(self._on_selection_canceled)
        self._sel_worker.finished.connect(self._on_selection_thread_done)
        self._sel_worker.start()

    def _on_selection_thread_done(self) -> None:
        worker = self.sender()
        current = worker is None or worker is self._sel_worker
        if worker is not None:
            worker.deleteLater()
        if not current:
            return  # an older run; the current one still owns the UI
        self._sel_worker = None
        if self._emb_thread is None:
            self.progress.setRange(0, 100)
            self._set_busy(False)

    def _on_selection_failed(self, msg: str) -> None:
        if not self._is_current(self._sel_worker):
            return
        self.lbl_status.setText("Smart Select failed.")
        QMessageBox.critical(self, "Smart Select failed", msg)

    def _on_selection_canceled(self) -> None:
        if not self._is_current(self._sel_worker):
            return
        self.lbl_status.setText("Canceled.")

    def _on_selection_ready(self, outcome: SmartSelectOutcome) -> None:
        if not self._is_current(self._sel_worker) or self._cancel_requested:
            return
        self._outcome = outcome
        self.batch_units = list(outcome.result.indices)
        self._unit_reason = dict(outcome.result.reasons)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self._autosave_clusters()
        self._refresh_table()
        info = outcome.result.info
        parts = [
            f"{len(outcome.units):,} {'frames' if self.frame_mode else 'images'} "
            f"in {outcome.n_clusters} clusters"
        ]
        if outcome.n_predicted:
            parts.append(f"{outcome.n_predicted:,} with predictions")
            parts.append(
                f"{outcome.n_error_scored:,} labeled compared to the model, "
                f"{int(info.get('n_errors', 0)):,} above the error threshold"
            )
        if info.get("suppressed"):
            parts.append(f"{int(info['suppressed']):,} near-duplicates skipped")
        self.lbl_status.setText("Done — " + "; ".join(parts) + ".")

    def _cancel(self) -> None:
        if self._busy:
            self._cancel_requested = True
            self.lbl_status.setText("Canceling…")
        if self._emb_worker is not None:
            self._emb_worker.cancel()
        if self._sel_worker is not None:
            self._sel_worker.cancel()

    def _stop_workers(self) -> None:
        """Cancel and detach background work so the dialog can close at once."""
        self._cancel()
        if self._emb_worker is not None:
            for sig in (
                self._emb_worker.finished,
                self._emb_worker.failed,
                self._emb_worker.progress,
                self._emb_worker.status,
            ):
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
        if self._emb_thread is not None:
            thread, worker = self._emb_thread, self._emb_worker
            # The worker stops at its next batch; the thread's event loop
            # then ends.  Keep both alive until it has.
            thread.quit()
            if not thread.wait(200):
                thread.finished.connect(worker.deleteLater)
                thread._smart_select_worker = worker  # keep the QObject alive
                _orphan_until_finished(thread)
            self._emb_thread = None
            self._emb_worker = None
        if self._sel_worker is not None:
            worker = self._sel_worker
            for sig in (
                worker.result,
                worker.error,
                worker.canceled,
                worker.status,
                worker.finished,
            ):
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
            if not worker.wait(200):
                _orphan_until_finished(worker)
            self._sel_worker = None

    # ── table ─────────────────────────────────────────────────────────────

    def _unit_name(self, unit: int) -> str:
        members = self._outcome.units[unit]
        name = self.image_paths[members[0]].name
        if len(members) > 1:
            name += f"  (+{len(members) - 1} on frame)"
        return name

    def _detail(self, unit: int, reason: str) -> str:
        out = self._outcome
        info = out.result.info
        cid = int(out.unit_clusters[unit])
        cov = info.get("coverage", {}).get(cid, 0.0)
        if reason == REASON_ERROR:
            src = info.get("error_sources", {}).get(unit)
            if src is not None:
                sev = info.get("error_severity", {}).get(src)
                sev_txt = f", error {sev:.2f}" if sev is not None else ""
                return f"looks like {self._unit_name(src)}{sev_txt}"
            return "looks like a model error"
        if reason == REASON_UNCERTAINTY:
            u = info.get("uncertainty", {}).get(unit)
            if u is not None:
                return f"mean keypoint confidence {1.0 - u:.2f}"
            return "low keypoint confidence"
        if reason == REASON_RARE:
            size = int((out.unit_clusters == cid).sum())
            return f"cluster of {size}, {cov:.0%} labeled"
        if reason == REASON_DIVERSITY:
            return "covers frames far from any label"
        if reason == REASON_REPRESENTATIVE:
            return f"dense cluster, {cov:.0%} labeled"
        if reason == REASON_AUDIT:
            return "random spot check"
        if reason == REASON_EXPLORER:
            return "added by hand"
        return ""

    def _refresh_table(self) -> None:
        self.table.setRowCount(0)
        if self._outcome is None:
            self._set_busy(self._busy)
            return
        self.table.setRowCount(len(self.batch_units))
        for row, unit in enumerate(self.batch_units):
            reason = self._unit_reason.get(unit, "")
            cells = [
                str(row + 1),
                self._unit_name(unit),
                str(int(self._outcome.unit_clusters[unit])),
                _REASON_TEXT.get(reason, reason),
                self._detail(unit, reason),
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col == 1:
                    item.setToolTip(str(self.image_paths[self._outcome.units[unit][0]]))
                    item.setData(Qt.UserRole, unit)
                self.table.setItem(row, col, item)
        self._update_summary()
        self._set_busy(self._busy)

    def _update_summary(self) -> None:
        if not self.batch_units:
            self.summary_label.setText("The batch is empty.")
            return
        counts: Dict[str, int] = {}
        for u in self.batch_units:
            r = self._unit_reason.get(u, "")
            counts[r] = counts.get(r, 0) + 1
        order = list(REASONS) + [REASON_EXPLORER]
        parts = [f"{_REASON_TEXT[r]} ×{counts[r]}" for r in order if counts.get(r)]
        n_inst = sum(len(self._outcome.units[u]) for u in self.batch_units)
        unit_word = "frames" if self.frame_mode else "images"
        extra = f" ({n_inst} individuals)" if self.frame_mode else ""
        self.summary_label.setText(
            f"<b>{len(self.batch_units)} {unit_word}{extra}</b> — " + " · ".join(parts)
        )

    def _selected_rows(self) -> List[int]:
        return sorted({i.row() for i in self.table.selectedIndexes()})

    def _on_table_selection(self) -> None:
        rows = self._selected_rows()
        if not rows or self._outcome is None:
            return
        unit = self.batch_units[rows[0]]
        path = self.image_paths[self._outcome.units[unit][0]]
        pix = QPixmap(str(path))
        if pix.isNull():
            self.preview.setText(f"Could not load\n{path.name}")
            return
        self.preview.setPixmap(
            pix.scaled(
                max(8, self.preview.width() - 8),
                max(8, self.preview.height() - 8),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def _remove_selected_rows(self) -> None:
        if self._busy:
            return
        rows = self._selected_rows()
        if not rows:
            return
        drop = {self.batch_units[r] for r in rows}
        self.batch_units = [u for u in self.batch_units if u not in drop]
        self._refresh_table()
        if self.batch_units:
            self.table.selectRow(min(rows[0], len(self.batch_units) - 1))

    # ── explorer / CSV ────────────────────────────────────────────────────

    def _open_explorer(self) -> None:
        if self._busy or self._outcome is None:
            return
        from .exploration import EmbeddingExplorerDialog

        out = self._outcome
        dialog = EmbeddingExplorerDialog(
            embeddings=out.unit_embeddings,
            image_paths=[self.image_paths[m[0]] for m in out.units],
            cluster_ids=out.unit_clusters,
            labeled_mask=list(out.unit_labeled),
            initial_selection=list(self.batch_units),
            accept_text="Use Selection as Batch",
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        chosen = list(dict.fromkeys(int(i) for i in dialog.selected_indices))
        chosen_set = set(chosen)
        kept = [u for u in self.batch_units if u in chosen_set]
        added = [u for u in chosen if u not in set(kept)]
        for u in added:
            self._unit_reason[u] = REASON_EXPLORER
        self.batch_units = kept + added
        self._refresh_table()

    def _write_clusters_csv(self, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "cluster_id"])
            for i, path in enumerate(self.image_paths):
                w.writerow([str(path), int(self._outcome.instance_clusters[i])])

    def _autosave_clusters(self) -> None:
        """Keep ``clusters.csv`` (cluster sort, cluster-stratified split) current."""
        if self._outcome is None:
            return
        try:
            self._write_clusters_csv(
                Path(self.project.out_root) / "posekit" / "clusters" / "clusters.csv"
            )
        except OSError:
            logger.warning("Could not write clusters.csv", exc_info=True)

    def _save_csv(self) -> None:
        if self._outcome is None:
            return
        default = Path(self.project.out_root) / "posekit" / "clusters" / "clusters.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save cluster assignments", str(default), "CSV (*.csv)"
        )
        if not path:
            return
        try:
            self._write_clusters_csv(Path(path))
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        QMessageBox.information(self, "Saved", f"Wrote cluster CSV:\n{path}")

    # ── finish ────────────────────────────────────────────────────────────

    def batch_instances(self) -> List[int]:
        """Instance indices of the batch in order (every individual of a frame)."""
        if self._outcome is None:
            return []
        out: List[int] = []
        seen = set()
        for u in self.batch_units:
            for i in self._outcome.units[u]:
                if i not in seen:
                    seen.add(i)
                    out.append(i)
        return out

    def _finish(self, start: bool) -> None:
        if self._busy:
            return
        self.selected_indices = self.batch_instances()
        if not self.selected_indices:
            return
        self.start_labeling = bool(start)
        self._did_add = True
        self.accept()
