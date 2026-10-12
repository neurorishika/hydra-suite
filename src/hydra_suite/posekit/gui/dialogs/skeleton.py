"""Skeleton editor dialog for defining keypoint names and edge topology."""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from hydra_suite.core.pose_flip import parse_flip_pairs, remap_pairs_by_name
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811

from ..panels.flip_pairs_panel import FlipPairsPanel


class SkeletonEditorDialog(QDialog):
    """Dialog to edit keypoint names and skeleton edge topology."""

    def __init__(
        self,
        keypoint_names: List[str],
        edges: List[Tuple[int, int]],
        parent=None,
        default_dir: Optional[Path] = None,
        flip_pairs: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Skeleton Editor")
        self.setMinimumSize(QSize(720, 420))

        self.kpt_names = list(keypoint_names)
        self.edges = list(edges)
        self.default_dir = default_dir

        # Main layout
        outer = QVBoxLayout(self)

        # Top section with keypoints and edges side-by-side
        root = QHBoxLayout()

        left = QVBoxLayout()
        left.addWidget(QLabel("Keypoints (order matters)"))
        self.kpt_table = QTableWidget(len(self.kpt_names), 2)
        self.kpt_table.setHorizontalHeaderLabels(["Index", "Name"])
        self.kpt_table.verticalHeader().setVisible(False)
        self.kpt_table.setColumnWidth(0, 60)
        self.kpt_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.kpt_table.setSelectionMode(QTableWidget.SingleSelection)
        for i, name in enumerate(self.kpt_names):
            self.kpt_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.kpt_table.item(i, 0).setFlags(Qt.ItemIsEnabled)
            self.kpt_table.setItem(i, 1, QTableWidgetItem(name))
        left.addWidget(self.kpt_table)

        kp_btns = QHBoxLayout()
        self.btn_kp_add = QPushButton("Add")
        self.btn_kp_del = QPushButton("Remove")
        self.btn_kp_ren = QPushButton("Rename")
        self.btn_kp_up = QPushButton("Up")
        self.btn_kp_down = QPushButton("Down")
        kp_btns.addWidget(self.btn_kp_add)
        kp_btns.addWidget(self.btn_kp_del)
        kp_btns.addWidget(self.btn_kp_ren)
        kp_btns.addWidget(self.btn_kp_up)
        kp_btns.addWidget(self.btn_kp_down)
        left.addLayout(kp_btns)

        btn_row = QHBoxLayout()
        self.btn_chain = QPushButton("Make Chain (i→i+1)")
        self.btn_clear_edges = QPushButton("Clear Edges")
        btn_row.addWidget(self.btn_chain)
        btn_row.addWidget(self.btn_clear_edges)
        left.addLayout(btn_row)

        io_row = QHBoxLayout()
        self.btn_load = QPushButton("Load Skeleton…")
        self.btn_save = QPushButton("Save Skeleton…")
        io_row.addWidget(self.btn_load)
        io_row.addWidget(self.btn_save)
        left.addLayout(io_row)

        right = QVBoxLayout()
        right.addWidget(QLabel("Skeleton edges (0-based indices)"))
        self.edge_table = QTableWidget(0, 2)
        self.edge_table.setHorizontalHeaderLabels(["A", "B"])
        self.edge_table.verticalHeader().setVisible(False)
        self.edge_table.setColumnWidth(0, 60)
        self.edge_table.setColumnWidth(1, 60)
        right.addWidget(self.edge_table)

        add_row = QHBoxLayout()
        self.a_spin = QSpinBox()
        self.b_spin = QSpinBox()
        mx = max(0, len(self.kpt_names) - 1)
        self.a_spin.setRange(0, mx)
        self.b_spin.setRange(0, mx)
        self.btn_add = QPushButton("Add Edge")
        self.btn_del = QPushButton("Delete Selected")
        add_row.addWidget(QLabel("Node A"))
        add_row.addWidget(self.a_spin)
        add_row.addWidget(QLabel("Node B"))
        add_row.addWidget(self.b_spin)
        add_row.addWidget(self.btn_add)
        add_row.addWidget(self.btn_del)
        right.addLayout(add_row)

        self.flip_panel = FlipPairsPanel(self.kpt_names, flip_pairs or [], self)
        right.addWidget(self.flip_panel)

        root.addLayout(left, 3)
        root.addLayout(right, 2)

        outer.addLayout(root)

        # Bottom buttons
        bottom = QHBoxLayout()
        self.btn_ok = QPushButton("OK")
        self.btn_cancel = QPushButton("Cancel")
        bottom.addStretch(1)
        bottom.addWidget(self.btn_ok)
        bottom.addWidget(self.btn_cancel)

        outer.addLayout(bottom)

        self.btn_add.clicked.connect(self._add_edge)
        self.btn_del.clicked.connect(self._del_edges)
        self.btn_chain.clicked.connect(self._make_chain)
        self.btn_clear_edges.clicked.connect(self._clear_edges)
        self.btn_load.clicked.connect(self._load_config)
        self.btn_save.clicked.connect(self._save_config)
        self.btn_kp_add.clicked.connect(self._kp_add)
        self.btn_kp_del.clicked.connect(self._kp_del)
        self.btn_kp_ren.clicked.connect(self._kp_rename)
        self.btn_kp_up.clicked.connect(lambda: self._kp_move(-1))
        self.btn_kp_down.clicked.connect(lambda: self._kp_move(1))
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self.kpt_table.itemChanged.connect(self._on_kpt_item_changed)

        self._refresh_edges()

    def _refresh_edges(self):
        self.edge_table.setRowCount(len(self.edges))
        for r, (a, b) in enumerate(self.edges):
            self.edge_table.setItem(r, 0, QTableWidgetItem(str(a)))
            self.edge_table.setItem(r, 1, QTableWidgetItem(str(b)))

    def _current_names(self) -> List[str]:
        names = []
        for i in range(self.kpt_table.rowCount()):
            item = self.kpt_table.item(i, 1)
            names.append(item.text().strip() if item else f"kp{i + 1}")
        return names

    def _on_kpt_item_changed(self, item) -> None:
        if item is not None and item.column() == 1:
            self.flip_panel.set_names(self._current_names())

    def _refresh_kpt_indices(self):
        for i in range(self.kpt_table.rowCount()):
            idx_item = self.kpt_table.item(i, 0)
            if idx_item is None:
                idx_item = QTableWidgetItem(str(i))
                self.kpt_table.setItem(i, 0, idx_item)
            idx_item.setText(str(i))
            idx_item.setFlags(Qt.ItemIsEnabled)

    def _update_kpt_ranges(self):
        mx = max(0, self.kpt_table.rowCount() - 1)
        self.a_spin.setRange(0, mx)
        self.b_spin.setRange(0, mx)

    def _selected_kpt_row(self) -> int:
        rows = {i.row() for i in self.kpt_table.selectedIndexes()}
        return next(iter(rows), -1)

    def _kp_add(self):
        i = self.kpt_table.rowCount()
        self.kpt_table.insertRow(i)
        self.kpt_table.setItem(i, 0, QTableWidgetItem(str(i)))
        self.kpt_table.item(i, 0).setFlags(Qt.ItemIsEnabled)
        self.kpt_table.setItem(i, 1, QTableWidgetItem(f"kp{i + 1}"))
        self.kpt_table.setCurrentCell(i, 1)
        self._update_kpt_ranges()
        self.flip_panel.set_names(self._current_names())

    def _kp_del(self):
        r = self._selected_kpt_row()
        if r < 0:
            return
        self.kpt_table.removeRow(r)
        self.edges = self._remap_edges_on_remove(self.edges, r)
        self.flip_panel.keypoint_removed(r, self._current_names())
        self._refresh_kpt_indices()
        self._update_kpt_ranges()
        self._refresh_edges()

    def _kp_rename(self):
        r = self._selected_kpt_row()
        if r < 0:
            return
        item = self.kpt_table.item(r, 1)
        current = item.text() if item else f"kp{r + 1}"
        name, ok = self._simple_text_prompt("Rename keypoint", "New name:", current)
        if ok and name.strip():
            if item is None:
                item = QTableWidgetItem(name.strip())
                self.kpt_table.setItem(r, 1, item)
            else:
                item.setText(name.strip())

    def _kp_move(self, delta: int):
        r = self._selected_kpt_row()
        if r < 0:
            return
        nr = r + delta
        if nr < 0 or nr >= self.kpt_table.rowCount():
            return

        name_item = self.kpt_table.takeItem(r, 1)
        if name_item is None:
            name_item = QTableWidgetItem(f"kp{r + 1}")
        self.kpt_table.removeRow(r)
        self.kpt_table.insertRow(nr)
        self.kpt_table.setItem(nr, 0, QTableWidgetItem(str(nr)))
        self.kpt_table.item(nr, 0).setFlags(Qt.ItemIsEnabled)
        self.kpt_table.setItem(nr, 1, name_item)
        self.kpt_table.setCurrentCell(nr, 1)

        self.edges = self._remap_edges_on_move(self.edges, r, nr)
        self.flip_panel.keypoint_moved(r, nr, self._current_names())
        self._refresh_kpt_indices()
        self._refresh_edges()

    @staticmethod
    def _remap_edges_on_remove(
        edges: List[Tuple[int, int]], removed: int
    ) -> List[Tuple[int, int]]:
        out = []
        for a, b in edges:
            if a == removed or b == removed:
                continue
            na = a - 1 if a > removed else a
            nb = b - 1 if b > removed else b
            if na != nb:
                out.append((na, nb))
        return out

    @staticmethod
    def _remap_edges_on_move(
        edges: List[Tuple[int, int]], old: int, new: int
    ) -> List[Tuple[int, int]]:
        if old == new:
            return list(edges)

        def remap_idx(i: int) -> int:
            """remap_idx method documentation."""
            if i == old:
                return new
            if new > old and old < i <= new:
                return i - 1
            if new < old and new <= i < old:
                return i + 1
            return i

        out = []
        for a, b in edges:
            na = remap_idx(a)
            nb = remap_idx(b)
            if na != nb:
                out.append((na, nb))
        return out

    def _add_edge(self):
        a = int(self.a_spin.value())
        b = int(self.b_spin.value())
        if a == b:
            return
        if (a, b) not in self.edges and (b, a) not in self.edges:
            self.edges.append((a, b))
            self._refresh_edges()

    def _del_edges(self):
        rows = sorted(
            {i.row() for i in self.edge_table.selectedIndexes()}, reverse=True
        )
        for r in rows:
            if 0 <= r < len(self.edges):
                self.edges.pop(r)
        self._refresh_edges()

    def _make_chain(self):
        n = self.kpt_table.rowCount()
        self.edges = [(i, i + 1) for i in range(max(0, n - 1))]
        self._refresh_edges()

    def _clear_edges(self):
        self.edges = []
        self._refresh_edges()

    def _simple_text_prompt(
        self, title: str, label: str, default: str
    ) -> Tuple[str, bool]:
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        lay = QFormLayout(dlg)
        le = QLineEdit(default)
        lay.addRow(label, le)
        row = QHBoxLayout()
        ok = QPushButton("OK")
        cancel = QPushButton("Cancel")
        row.addStretch(1)
        row.addWidget(ok)
        row.addWidget(cancel)
        lay.addRow(row)
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        if dlg.exec() == QDialog.Accepted:
            return le.text(), True
        return default, False

    def _pick_skeleton_path(self, save: bool) -> Optional[Path]:
        start_dir = str(self.default_dir) if self.default_dir else os.getcwd()
        if save:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Skeleton",
                os.path.join(start_dir, "skeleton.json"),
                "Skeleton JSON (*.json)",
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Skeleton",
                start_dir,
                "Skeleton JSON (*.json)",
            )
        return Path(path) if path else None

    def _apply_config(
        self,
        names: List[str],
        edges: List[Tuple[int, int]],
        flip_pairs: Optional[List[Tuple[int, int]]] = None,
    ):
        if not names:
            return
        if len(names) != self.kpt_table.rowCount():
            resp = QMessageBox.question(
                self,
                "Replace keypoints?",
                f"Skeleton has {len(names)} keypoints but editor has {self.kpt_table.rowCount()}\n"
                "Replace keypoints with loaded list?",
            )
            if resp != QMessageBox.Yes:
                return
            self.kpt_table.setRowCount(len(names))

        old_names = self._current_names()
        old_pairs = self.flip_panel.pairs()
        self.kpt_names = list(names)
        for i, name in enumerate(self.kpt_names):
            self.kpt_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.kpt_table.item(i, 0).setFlags(Qt.ItemIsEnabled)
            self.kpt_table.setItem(i, 1, QTableWidgetItem(name))

        self._update_kpt_ranges()

        k = len(self.kpt_names)
        self.edges = [
            (a, b) for (a, b) in edges if 0 <= a < k and 0 <= b < k and a != b
        ]
        self._refresh_edges()
        self.flip_panel.set_names(self.kpt_names)
        if flip_pairs is None:
            # A file saved without pairs: keep the current pairs only for
            # keypoints it still has, matched by name, never by position.
            flip_pairs = remap_pairs_by_name(old_pairs, old_names, self.kpt_names)
        self.flip_panel.set_pairs(flip_pairs)

    def _load_config(self):
        path = self._pick_skeleton_path(save=False)
        if path is None:
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", f"Failed to read file:\n{exc}")
            return

        names = data.get("keypoint_names") or data.get("keypoints")
        edges = data.get("skeleton_edges") or data.get("edges")
        if not isinstance(names, list) or not isinstance(edges, list):
            QMessageBox.critical(
                self, "Invalid file", "Missing keypoint_names or edges."
            )
            return

        parsed_edges = []
        for e in edges:
            if isinstance(e, (list, tuple)) and len(e) == 2:
                try:
                    parsed_edges.append((int(e[0]), int(e[1])))
                except Exception:
                    continue

        self._apply_config(
            [str(n) for n in names],
            parsed_edges,
            parse_flip_pairs(data) if "flip_pairs" in data else None,
        )

    def _save_config(self):
        path = self._pick_skeleton_path(save=True)
        if path is None:
            return
        names, edges = self.get_result()
        data = {
            "keypoint_names": names,
            "skeleton_edges": [[a, b] for a, b in edges],
            "flip_pairs": [[a, b] for a, b in self.get_flip_pairs()],
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"Failed to write file:\n{exc}")

    def get_result(self) -> Tuple[List[str], List[Tuple[int, int]]]:
        """get_result method documentation."""
        out_names = []
        for i in range(self.kpt_table.rowCount()):
            item = self.kpt_table.item(i, 1)
            out_names.append(item.text().strip() if item else f"kp{i + 1}")
        # Return at least the defaults if empty
        if not out_names:
            out_names = ["kp1", "kp2"]
        return out_names, list(self.edges)

    def get_flip_pairs(self) -> List[Tuple[int, int]]:
        """Left/right keypoint pairs, in the index space of :meth:`get_result`."""
        return self.flip_panel.pairs()
