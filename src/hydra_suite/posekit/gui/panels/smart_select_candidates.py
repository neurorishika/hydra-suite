"""Smart Select candidates: the ordered batch, why each frame is in it, a preview.

A view: the dialog owns the batch and calls :meth:`show_batch` after every
change; Delete / *Remove from batch* emit :attr:`remove_requested`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Collection, Dict, List, Optional, Sequence

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.posekit.core.selection import (
    REASON_AUDIT,
    REASON_DIVERSITY,
    REASON_ERROR,
    REASON_LABELS,
    REASON_RARE,
    REASON_REPRESENTATIVE,
    REASON_UNCERTAINTY,
    REASONS,
)

REASON_EXPLORER = "explorer"
REASON_TEXT = dict(REASON_LABELS, **{REASON_EXPLORER: "Picked in Explorer"})

# Widgets the dialog exposes under the same names.
WIDGET_NAMES = [
    "summary_label",
    "table",
    "preview",
    "btn_remove",
    "btn_explorer",
    "btn_save_csv",
]


class SmartSelectCandidatesPanel(QWidget):
    """Ordered candidate table with reasons, summary and image preview."""

    remove_requested = Signal()

    def __init__(self, image_paths: Sequence[Path], frame_mode: bool, parent=None):
        super().__init__(parent)
        self.image_paths = list(image_paths)
        self.frame_mode = bool(frame_mode)
        self._outcome = None
        self._batch: List[int] = []
        self._unreadable: Collection[int] = ()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
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

        buttons = QHBoxLayout()
        self.btn_remove = QPushButton("Remove from batch")
        self.btn_remove.setToolTip("Drop the selected candidates (Delete).")
        self.btn_remove.clicked.connect(self.remove_requested.emit)
        self.btn_explorer = QPushButton("Embedding Explorer…")
        self.btn_explorer.setToolTip(
            "See every frame on a UMAP map with the batch highlighted; click "
            "points to add or drop them."
        )
        self.btn_save_csv = QPushButton("Save clusters CSV…")
        buttons.addWidget(self.btn_remove)
        buttons.addWidget(self.btn_explorer)
        buttons.addWidget(self.btn_save_csv)
        buttons.addStretch(1)
        left_l.addLayout(buttons)
        split.addWidget(left)

        self.preview = QLabel("Select a candidate to preview it.")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(240, 240)
        self.preview.setWordWrap(True)
        self.preview.setStyleSheet("border: 1px solid #3e3e42; border-radius: 4px;")
        split.addWidget(self.preview)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        outer.addWidget(split)

        for key in (QKeySequence.Delete, QKeySequence(Qt.Key_Backspace)):
            sc = QShortcut(key, self.table)
            sc.setContext(Qt.WidgetShortcut)
            sc.activated.connect(self.remove_requested.emit)

    # ── names ────────────────────────────────────────────────────────────

    def unit_path(self, unit: int) -> Path:
        """A readable image of the unit (its first one when none is)."""
        members = self._outcome.units[unit]
        first = next((i for i in members if i not in self._unreadable), members[0])
        return self.image_paths[first]

    def _readable(self, unit: int) -> List[int]:
        return [i for i in self._outcome.units[unit] if i not in self._unreadable]

    def unit_name(self, unit: int) -> str:
        name = self.unit_path(unit).name
        n = len(self._readable(unit))
        if n > 1:
            name += f"  (+{n - 1} on frame)"
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
                return f"looks like {self.unit_name(src)}{sev_txt}"
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

    # ── rendering ────────────────────────────────────────────────────────

    def show_batch(
        self,
        outcome,
        batch_units: Sequence[int],
        reasons: Dict[int, str],
        unreadable: Collection[int] = (),
        labeled_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Render ``batch_units`` (in order) of ``outcome``."""
        self._outcome = outcome
        self._batch = list(batch_units)
        self._unreadable = set(unreadable)
        self.table.setRowCount(0)
        if outcome is None:
            return
        self.table.setRowCount(len(self._batch))
        for row, unit in enumerate(self._batch):
            reason = reasons.get(unit, "")
            detail = self._detail(unit, reason)
            members = outcome.units[unit]
            if labeled_mask is not None and len(members) > 1:
                n_lab = int(np.asarray(labeled_mask)[members].sum())
                if n_lab:
                    detail += f" ({n_lab} of {len(members)} already labeled)"
            cells = [
                str(row + 1),
                self.unit_name(unit),
                str(int(outcome.unit_clusters[unit])),
                REASON_TEXT.get(reason, reason),
                detail,
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col == 1:
                    item.setToolTip(str(self.unit_path(unit)))
                    item.setData(Qt.UserRole, unit)
                self.table.setItem(row, col, item)
        self._show_summary(reasons)

    def _show_summary(self, reasons: Dict[int, str]) -> None:
        if not self._batch:
            self.summary_label.setText("The batch is empty.")
            return
        counts: Dict[str, int] = {}
        for u in self._batch:
            r = reasons.get(u, "")
            counts[r] = counts.get(r, 0) + 1
        order = list(REASONS) + [REASON_EXPLORER]
        parts = [f"{REASON_TEXT[r]} ×{counts[r]}" for r in order if counts.get(r)]
        n_inst = sum(len(self._readable(u)) for u in self._batch)
        unit_word = "frames" if self.frame_mode else "images"
        extra = f" ({n_inst} individuals)" if self.frame_mode else ""
        self.summary_label.setText(
            f"<b>{len(self._batch)} {unit_word}{extra}</b> — " + " · ".join(parts)
        )

    def selected_rows(self) -> List[int]:
        return sorted({i.row() for i in self.table.selectedIndexes()})

    def _on_table_selection(self) -> None:
        rows = self.selected_rows()
        if not rows or self._outcome is None or rows[0] >= len(self._batch):
            return
        path = self.unit_path(self._batch[rows[0]])
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
