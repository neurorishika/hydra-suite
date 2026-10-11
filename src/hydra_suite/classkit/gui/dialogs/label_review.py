"""Label Review: page through every image of a label and remove mislabels.

A view over :class:`~hydra_suite.classkit.core.al.label_review.LabelReviewModel`.
Click tiles (or press their keys) to select them, then *Remove label* clears
the selection's labels in one undoable step.  Removals are committed right
away through ``labels_removed``; undo emits ``labels_restored``.

Only tiles whose image is on screen can be selected, so nothing is removed
unseen.
"""

from __future__ import annotations

import html
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Sequence, Set

import numpy as np
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.classkit.core.al.label_review import LabelReviewModel
from hydra_suite.widgets.dialogs import BaseDialog

from ..widgets.image_prefetch import ImagePrefetcher
from ..widgets.image_tile import ImageTile, load_scaled
from .grid_balance import KEYS_3X3, KEYS_4X4

GRID_SIZES = {"3 × 3": (3, 3), "4 × 4": (4, 4), "6 × 4": (6, 4)}
SOURCE_FILTERS = {
    "All labels": "all",
    "Human-verified": "human",
    "Machine (unverified)": "machine",
}
_PIXMAP_CACHE_LIMIT = 512


class LabelReviewDialog(BaseDialog):
    """Browse labels, select mislabeled images, clear their labels."""

    labels_removed = Signal(object)  # list[int]
    labels_restored = Signal(object)  # list[tuple[int, str]]

    def __init__(
        self,
        model: LabelReviewModel,
        image_paths: Sequence[Path | str],
        *,
        initial_label: Optional[str] = None,
        tile_edge: int = 150,
        parent=None,
    ) -> None:
        super().__init__("Review Labels", parent, QDialogButtonBox.Close)
        self.model = model
        self._paths = [str(p) for p in image_paths]
        self._tile_edge = int(tile_edge)
        self._pixmaps: "OrderedDict[str, QPixmap]" = OrderedDict()
        self._failed: Set[str] = set()
        self._tiles: List[ImageTile] = []
        self._tile_shortcuts: List[QShortcut] = []
        self._items: List[int] = []
        self._selected: Set[int] = set()
        self._label: Optional[str] = None
        self._page = 0
        self._undo_failed = False
        self._unsaved = False
        self._cols, self._rows = GRID_SIZES["4 × 4"]
        self._prefetcher = ImagePrefetcher(self)
        self._prefetcher.image_ready.connect(self._on_image_ready)
        self._prefetcher.image_failed.connect(self._on_image_failed)
        for btn in self._buttons.buttons():
            # Nothing may hold focus: Space/Enter would otherwise hit Close.
            btn.setAutoDefault(False)
            btn.setDefault(False)
            btn.setFocusPolicy(Qt.NoFocus)
        self.setFocusPolicy(Qt.StrongFocus)

        root = QWidget(self)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)

        # Left: filters and the label list.
        side = QVBoxLayout()
        side.setSpacing(6)
        self.source_combo = QComboBox()
        self.source_combo.addItems(list(SOURCE_FILTERS))
        self.source_combo.setFocusPolicy(Qt.NoFocus)
        self.source_combo.currentTextChanged.connect(self._on_filter_changed)
        side.addWidget(self.source_combo)
        self.disagree_check = QCheckBox("Model disagrees only")
        self.disagree_check.setFocusPolicy(Qt.NoFocus)
        self.disagree_check.setEnabled(model.predicted is not None)
        if model.predicted is None:
            self.disagree_check.setToolTip("Needs a trained model's predictions.")
        self.disagree_check.toggled.connect(self._on_filter_changed)
        side.addWidget(self.disagree_check)
        self.label_list = QListWidget()
        self.label_list.setFocusPolicy(Qt.NoFocus)
        self.label_list.setMinimumWidth(220)
        self.label_list.currentItemChanged.connect(self._on_label_changed)
        side.addWidget(self.label_list, 1)
        outer.addLayout(side)

        # Right: header, grid, controls.
        main = QVBoxLayout()
        main.setSpacing(8)
        head = QHBoxLayout()
        self.title_label = QLabel()
        self.title_label.setTextFormat(Qt.RichText)
        # Long label names are elided instead of widening the dialog.
        self.title_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.title_label.setMinimumWidth(0)
        head.addWidget(self.title_label, 1)
        self.page_label = QLabel()
        self.page_label.setStyleSheet("color: #aaa;")
        head.addWidget(self.page_label)
        main.addLayout(head)
        self.order_label = QLabel()
        self.order_label.setStyleSheet("color: #888; font-size: 11px;")
        main.addWidget(self.order_label)

        self.grid_host = QWidget(root)
        self.grid_layout = QGridLayout(self.grid_host)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(6)
        self.grid_host.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        main.addWidget(self.grid_host, 0, Qt.AlignHCenter)

        nav = QHBoxLayout()
        nav.addWidget(QLabel("Grid:"))
        self.size_combo = QComboBox()
        self.size_combo.addItems(list(GRID_SIZES))
        self.size_combo.setCurrentText("4 × 4")
        self.size_combo.setFocusPolicy(Qt.NoFocus)
        self.size_combo.currentTextChanged.connect(self._on_size_changed)
        nav.addWidget(self.size_combo)
        nav.addStretch(1)
        self.prev_btn = self._button("◀  Prev  (PgUp)", self.prev_page)
        self.next_btn = self._button("Next  ▶  (PgDn)", self.next_page)
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        main.addLayout(nav)

        actions = QHBoxLayout()
        self.select_page_btn = self._button("Select page  (Ctrl+A)", self.select_page)
        self.clear_sel_btn = self._button("Clear selection", self.clear_selection)
        actions.addWidget(self.select_page_btn)
        actions.addWidget(self.clear_sel_btn)
        actions.addStretch(1)
        self.undo_btn = self._button("↶  Undo  (Ctrl+Z)", self.undo)
        self.remove_btn = self._button("✕  Remove label  (Del)", self.remove_selected)
        self.remove_btn.setStyleSheet(
            "QPushButton { background: #b3261e; font-weight: bold; }"
            "QPushButton:disabled { background: #4a2a28; color: #888; }"
        )
        actions.addWidget(self.undo_btn)
        actions.addWidget(self.remove_btn)
        main.addLayout(actions)

        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        main.addWidget(self.status_label)
        outer.addLayout(main, 1)
        self.add_content(root)
        for combo in (self.source_combo, self.size_combo):
            combo.installEventFilter(self)  # no silent changes from the wheel

        for seq, slot in (
            ("Delete", self.remove_selected),
            ("PgDown", self.next_page),
            ("PgUp", self.prev_page),
            ("Right", self.next_page),
            ("Left", self.prev_page),
            ("Down", lambda: self._step_label(1)),
            ("Up", lambda: self._step_label(-1)),
            (QKeySequence.SelectAll, self.select_page),
            # Backspace is the Mac "delete" key, so it removes too; undo is
            # Ctrl/Cmd+Z only.
            ("Backspace", self.remove_selected),
            (QKeySequence.Undo, self.undo),
        ):
            self._shortcut(QKeySequence(seq), slot)

        self._build_grid()
        self._fill_label_list(initial_label)

    # ── actions ──────────────────────────────────────────────────────────
    def toggle_slot(self, slot: int) -> None:
        idx = self._index_at(slot)
        if idx is None or not self._visible(idx):
            return
        if idx in self._selected:
            self._selected.discard(idx)
        else:
            self._selected.add(idx)
        self._render()

    def select_page(self) -> None:
        page = [i for i in self._page_items() if self._visible(i)]
        if page and all(i in self._selected for i in page):
            self._selected.difference_update(page)
        else:
            self._selected.update(page)
        self._render()

    def clear_selection(self) -> None:
        self._selected.clear()
        self._render()

    def remove_selected(self) -> None:
        # Only what is on screen right now can be removed.
        chosen = [
            i for i in self._page_items() if i in self._selected and self._visible(i)
        ]
        removal = self.model.remove(chosen)
        self._selected.clear()
        if removal is None:
            self._set_status("Select images first: click them or press their keys.")
            return
        indices = [i for i, _ in removal.entries]
        self._unsaved = False
        self.labels_removed.emit(indices)
        n = len(indices)
        self._set_status(
            f"Removed the label from {n} image{'s' if n != 1 else ''} "
            + (
                "but it is not saved yet: the project database is busy "
                "(ClassKit keeps retrying)."
                if self._unsaved
                else "(now unlabeled)."
            )
        )
        self._refresh_items()

    def undo(self) -> None:
        removal = self.model.undo()
        if removal is None:
            self._set_status("Nothing to undo.")
            return
        self._undo_failed = False
        self.labels_restored.emit(list(removal.entries))
        if self._undo_failed:  # the host put the removal back
            return
        labels = {lab for _, lab in removal.entries}
        n = len(removal.entries)
        self._set_status(
            f"Restored {n} label{'s' if n != 1 else ''} ({', '.join(sorted(labels))})."
        )
        first, target = removal.entries[0]
        shown = set(
            self.model.items(target, self._source(), self.disagree_check.isChecked())
        )
        if any(i not in shown for i, _ in removal.entries):
            # The restored images are hidden by a filter: show everything.
            for w in (self.source_combo, self.disagree_check):
                w.blockSignals(True)
            self.source_combo.setCurrentText("All labels")
            self.disagree_check.setChecked(False)
            for w in (self.source_combo, self.disagree_check):
                w.blockSignals(False)
        self._fill_label_list(target)
        if first in self._items:  # jump to the page showing the restored image
            self._page = self._items.index(first) // self._page_size()
            self._render()

    def removal_unsaved(self) -> None:
        """The host could not save a removal yet (autosave retries)."""
        self._unsaved = True

    def restore_failed(self, entries) -> None:
        """The host could not save an undo: put the removal back."""
        self._undo_failed = True
        self.model.redo_removal(entries)
        self._set_status("Undo failed: the labels are still removed.")
        self._refresh_items()

    def next_page(self) -> None:
        if self._page + 1 < self._page_count():
            self._page += 1
            self._render()

    def prev_page(self) -> None:
        if self._page > 0:
            self._page -= 1
            self._render()

    # ── state ────────────────────────────────────────────────────────────
    def _source(self) -> str:
        return SOURCE_FILTERS[self.source_combo.currentText()]

    def _page_size(self) -> int:
        return self._cols * self._rows

    def _page_count(self) -> int:
        return max(1, -(-len(self._items) // self._page_size()))

    def _page_items(self) -> List[int]:
        start = self._page * self._page_size()
        return self._items[start : start + self._page_size()]

    def _index_at(self, slot: int) -> Optional[int]:
        page = self._page_items()
        return page[slot] if 0 <= slot < len(page) else None

    def _visible(self, idx: int) -> bool:
        """The user can see this image (or that it is unreadable)."""
        path = self._paths[idx]
        return path in self._pixmaps or path in self._failed

    def _fill_label_list(self, select: Optional[str] = None) -> None:
        summary = self.model.summary(self._source())
        self.label_list.blockSignals(True)
        self.label_list.clear()
        row_to_select = 0
        for row, (label, counts) in enumerate(summary.items()):
            item = QListWidgetItem()
            item.setData(Qt.UserRole, label)
            self._set_list_text(item, *counts)
            self.label_list.addItem(item)
            if label == (select or self._label):
                row_to_select = row
        if self.label_list.count():
            self.label_list.setCurrentRow(row_to_select)
        self.label_list.blockSignals(False)
        if self.label_list.count():
            self._on_label_changed(self.label_list.currentItem(), None)
        else:
            self._label = None
            self._refresh_items()

    def _refresh_items(self, keep_page: bool = True) -> None:
        if self._label is None:
            self._items = []
        else:
            self._items = self.model.items(
                self._label, self._source(), self.disagree_check.isChecked()
            )
        self._selected &= set(self._items)
        if not keep_page:
            self._page = 0
        self._page = min(self._page, self._page_count() - 1)
        self._update_list_counts()
        self._render()

    def _update_list_counts(self) -> None:
        summary = self.model.summary(self._source())
        for row in range(self.label_list.count()):
            item = self.label_list.item(row)
            label = item.data(Qt.UserRole)
            self._set_list_text(item, *summary.get(label, (0, 0)))

    def _set_list_text(self, item: QListWidgetItem, count: int, wrong: int) -> None:
        label = item.data(Qt.UserRole)
        shown = wrong if self.disagree_check.isChecked() else count
        text = f"{label}   {shown:,}"
        if self.model.predicted is not None and not self.disagree_check.isChecked():
            text += f"   ({wrong:,} ⚠)"
        item.setText(text)
        item.setToolTip(text)
        # Default role data keeps the stylesheet's text color for normal rows.
        item.setData(Qt.ForegroundRole, QColor("#777") if shown == 0 else None)

    def _on_label_changed(self, current, _previous) -> None:
        self._label = None if current is None else current.data(Qt.UserRole)
        self._selected.clear()
        self._refresh_items(keep_page=False)

    def _on_filter_changed(self, *_args) -> None:
        self._selected.clear()
        self._fill_label_list(self._label)

    def _step_label(self, delta: int) -> None:
        row = self.label_list.currentRow() + delta
        if 0 <= row < self.label_list.count():
            self.label_list.setCurrentRow(row)

    # ── view ─────────────────────────────────────────────────────────────
    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.setFocus()

    def _shortcut(self, seq, slot) -> QShortcut:
        sc = QShortcut(seq, self)
        sc.setAutoRepeat(False)  # holding a key must not act on unseen pages
        sc.activated.connect(slot)
        return sc

    def _button(self, text: str, slot) -> QPushButton:
        b = QPushButton(text)
        b.setAutoDefault(False)
        b.setFocusPolicy(Qt.NoFocus)
        b.clicked.connect(slot)
        return b

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _keys(self) -> List[str]:
        k = self._page_size()
        return KEYS_3X3 if k == 9 else (KEYS_4X4 if k == 16 else [])

    def _build_grid(self) -> None:
        for t in self._tiles:
            t.setParent(None)
            t.deleteLater()
        for sc in self._tile_shortcuts:
            sc.setEnabled(False)
            sc.setParent(None)
            sc.deleteLater()
        self._tiles, self._tile_shortcuts = [], []
        keys = self._keys()
        edge = (
            self._tile_edge if self._page_size() <= 16 else int(self._tile_edge * 0.8)
        )
        for slot in range(self._page_size()):
            key = keys[slot] if slot < len(keys) else ""
            tile = ImageTile(key, edge, self.grid_host)
            tile.clicked.connect(lambda _=False, s=slot: self.toggle_slot(s))
            self.grid_layout.addWidget(tile, slot // self._cols, slot % self._cols)
            self._tiles.append(tile)
            if key:
                self._tile_shortcuts.append(
                    self._shortcut(
                        QKeySequence(key), lambda s=slot: self.toggle_slot(s)
                    )
                )
        self.grid_host.adjustSize()

    def _on_size_changed(self, text: str) -> None:
        size = GRID_SIZES.get(text)
        if size is None or size == (self._cols, self._rows):
            return
        first = self._page * self._page_size()
        self._cols, self._rows = size
        self._page = first // self._page_size()  # keep the first visible image
        self._build_grid()
        self._render()
        self.layout().activate()
        self.adjustSize()

    def _pixmap(self, index: int) -> Optional[QPixmap]:
        path = self._paths[index]
        pm = self._pixmaps.get(path)
        if pm is not None:
            self._pixmaps.move_to_end(path)  # least-recently-used eviction
        elif path not in self._failed:
            edge = int(round(self._tile_edge * self.devicePixelRatioF()))
            self._prefetcher.request([path], lambda p: load_scaled(p, edge))
        return pm

    def _on_image_ready(self, path: str, image) -> None:
        while len(self._pixmaps) >= _PIXMAP_CACHE_LIMIT:
            self._pixmaps.popitem(last=False)
        pm = QPixmap.fromImage(image)
        pm.setDevicePixelRatio(self.devicePixelRatioF())
        self._pixmaps[path] = pm
        self._render_tiles()

    def _on_image_failed(self, path: str) -> None:
        self._failed.add(path)
        self._render_tiles()

    def _caption(self, idx: int):
        m = self.model
        tag = "M" if m.machine[idx] else ""
        if not m.has_model:
            return "", "#ddd", tag
        p = ""
        if m.agreement is not None and np.isfinite(m.agreement[idx]):
            p = f"P={m.agreement[idx]:.2f}"
        if m.disagrees(idx):
            # The model's guess first: small tiles elide the end.
            return f"→ {m.predicted[idx]}  {p}".rstrip(), "#ff8a65", tag
        return p, "#8bc34a", tag

    def _render(self) -> None:
        # Selection never outlives the page it was made on (page change,
        # grid resize, undo), so Del cannot remove anything off screen.
        self._selected &= set(self._page_items())
        if self._label is None:
            self.title_label.setText(
                "<span style='font-size:16px;'>No labeled images"
                + (" for this filter" if self.label_list.count() == 0 else "")
                + "</span>"
            )
        else:
            n = len(self._items)
            count = f"{n:,} image{'s' if n != 1 else ''}" if n else "nothing to show"
            if not n and self.disagree_check.isChecked():
                count = "the model agrees with every image"
            name = self._label if len(self._label) <= 60 else self._label[:59] + "…"
            self.title_label.setText(
                f"<span style='font-size:16px;'><b>{html.escape(name)}</b></span>"
                f"<span style='color:#aaa;'>  —  {count}</span>"
            )
            self.title_label.setToolTip(self._label)
        self.page_label.setText(f"Page {self._page + 1} / {self._page_count()}")
        if self.model.has_model:
            self.order_label.setText(
                "Sorted by model agreement, lowest first: likely mislabels are on "
                "the first pages.  M = unverified machine label."
            )
        else:
            self.order_label.setText(
                "No predictions loaded: images are in project order.  "
                "M = unverified machine label."
            )
        self.prev_btn.setEnabled(self._page > 0)
        self.next_btn.setEnabled(self._page + 1 < self._page_count())
        n_sel = len(self._selected)
        self.remove_btn.setEnabled(n_sel > 0)
        self.remove_btn.setText(
            f"✕  Remove label{f' from {n_sel}' if n_sel else ''}  (Del)"
        )
        self.clear_sel_btn.setEnabled(n_sel > 0)
        self.undo_btn.setEnabled(bool(self.model.history))
        self._render_tiles()

    def _render_tiles(self) -> None:
        page = self._page_items()
        for slot, tile in enumerate(self._tiles):
            idx = page[slot] if slot < len(page) else None
            if tile.property("review_index") != idx:
                # A press made on the previous image must not click this one.
                tile.setDown(False)
                tile.setProperty("review_index", idx)
            if idx is None:
                tile.set_image(None, True, " ")
                tile.set_caption("")
                tile.set_selected(False)
                tile.setToolTip("")
                continue
            failed = self._paths[idx] in self._failed
            tile.set_image(
                None if failed else self._pixmap(idx),
                False,
                "✕ unreadable" if failed else "",
            )
            tile.set_caption(*self._caption(idx))
            tile.set_selected(idx in self._selected)
            tile.setToolTip(Path(self._paths[idx]).name)

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Wheel and watched in (
            self.source_combo,
            self.size_combo,
        ):
            return True
        return super().eventFilter(watched, event)

    def done(self, result: int) -> None:  # noqa: D401 - Qt override
        self._prefetcher.shutdown(wait_ms=0)  # never freeze closing on slow disks
        super().done(result)
