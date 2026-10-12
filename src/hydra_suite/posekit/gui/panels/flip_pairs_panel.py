"""Left/right keypoint pairs for the Skeleton Editor.

Training mirrors images (left-right and up-down flips) and must then swap each
left keypoint with its right partner; this panel edits those pairs.  The
editor owns the keypoint list and tells the panel when names change or rows
move, so the pairs keep pointing at the same keypoints.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.core.pose_flip import (
    normalize_flip_pairs,
    remap_pairs_on_move,
    remap_pairs_on_remove,
    suggest_flip_pairs,
)

Pair = Tuple[int, int]


class FlipPairsPanel(QWidget):
    """Table of left/right pairs with add, delete and suggest-from-names."""

    def __init__(
        self, names: Sequence[str], pairs: Sequence[Pair], parent=None
    ) -> None:
        super().__init__(parent)
        self._names: List[str] = [str(n) for n in names]
        self._pairs: List[Pair] = normalize_flip_pairs(pairs, len(self._names))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Left/right pairs (swapped when training flips an image)")
        title.setWordWrap(True)
        layout.addWidget(title)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Keypoint", "Mirror partner"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        add_row = QHBoxLayout()
        self.combo_a = QComboBox()
        self.combo_b = QComboBox()
        self.btn_add = QPushButton("Add Pair")
        add_row.addWidget(self.combo_a, 1)
        add_row.addWidget(QLabel("↔"))
        add_row.addWidget(self.combo_b, 1)
        add_row.addWidget(self.btn_add)
        layout.addLayout(add_row)

        btn_row = QHBoxLayout()
        self.btn_suggest = QPushButton("Suggest from names")
        self.btn_suggest.setToolTip(
            "Pair keypoints whose names differ only by left/right "
            "(left_eye/right_eye, l_ear/r_ear, ...)."
        )
        self.btn_del = QPushButton("Delete Selected")
        btn_row.addWidget(self.btn_suggest)
        btn_row.addWidget(self.btn_del)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.btn_add.clicked.connect(self._add_pair)
        self.btn_del.clicked.connect(self._delete_selected)
        self.btn_suggest.clicked.connect(self._suggest)
        self._refresh()

    # ── editor hooks ─────────────────────────────────────────────────────

    def set_names(self, names: Sequence[str]) -> None:
        """Keypoint names changed (rename, add, load): relabel, drop stale pairs."""
        self._names = [str(n) for n in names]
        self._pairs = normalize_flip_pairs(self._pairs, len(self._names))
        self._refresh()

    def keypoint_removed(self, row: int, names: Sequence[str]) -> None:
        self._pairs = remap_pairs_on_remove(self._pairs, row)
        self.set_names(names)

    def keypoint_moved(self, old: int, new: int, names: Sequence[str]) -> None:
        self._pairs = remap_pairs_on_move(self._pairs, old, new)
        self.set_names(names)

    def set_pairs(self, pairs: Sequence[Pair]) -> None:
        self._pairs = normalize_flip_pairs(pairs, len(self._names))
        self._refresh()

    def pairs(self) -> List[Pair]:
        return list(self._pairs)

    # ── actions ──────────────────────────────────────────────────────────

    def _add_pair(self) -> None:
        a, b = self.combo_a.currentIndex(), self.combo_b.currentIndex()
        if a < 0 or b < 0 or a == b:
            self.status.setText("Pick two different keypoints.")
            return
        used = {i for pair in self._pairs for i in pair}
        if a in used or b in used:
            self.status.setText("Each keypoint can be in only one pair.")
            return
        self._pairs.append((min(a, b), max(a, b)))
        self._refresh()

    def _delete_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self._pairs):
                self._pairs.pop(r)
        self._refresh()

    def _suggest(self) -> None:
        found = suggest_flip_pairs(self._names)
        if not found:
            self.status.setText("No left/right names found.")
            return
        # Keep the existing pairs; add suggestions for keypoints still unpaired.
        self._pairs = normalize_flip_pairs(self._pairs + found, len(self._names))
        self._refresh()

    # ── rendering ────────────────────────────────────────────────────────

    def _label(self, i: int) -> str:
        name = self._names[i] if 0 <= i < len(self._names) else "?"
        return f"{i}: {name}"

    def _refresh(self) -> None:
        for combo in (self.combo_a, self.combo_b):
            current = combo.currentIndex()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems([self._label(i) for i in range(len(self._names))])
            if 0 <= current < combo.count():
                combo.setCurrentIndex(current)
            combo.blockSignals(False)
        self.table.setRowCount(len(self._pairs))
        for r, (a, b) in enumerate(self._pairs):
            self.table.setItem(r, 0, QTableWidgetItem(self._label(a)))
            self.table.setItem(r, 1, QTableWidgetItem(self._label(b)))
        if self._pairs:
            self.status.setText(f"{len(self._pairs)} pair(s).")
        else:
            self.status.setText(
                "No pairs: flips keep every keypoint in place. Add pairs for "
                "keypoints that have a left and a right side."
            )
