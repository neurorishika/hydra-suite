"""CAPTCHA-style grid balancer: click every image of the shown combination.

A thin view over :class:`~hydra_suite.classkit.core.al.grid_balance.GridBalanceSession`.
Every click is committed immediately through ``label_applied`` (undo emits the
label being cleared), so closing the dialog never loses work.

Guiding rule: an image is only labeled if the user could have seen it.  Tiles
that are still loading, were refilled a moment ago (double-click, key
autorepeat) or belong to a combination that just switched ignore input.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Sequence, Set

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.classkit.core.al.grid_balance import GridBalanceSession
from hydra_suite.widgets.dialogs import BaseDialog

from ..widgets.image_prefetch import ImagePrefetcher
from ..widgets.image_tile import ImageTile as _Tile
from ..widgets.image_tile import load_scaled as _load_scaled

# Tile keys in reading order; the 4x4 layout uses the left-hand key block.
KEYS_3X3 = list("123456789")
KEYS_4X4 = list("1234QWERASDFZXCV")
GRID_SIZES = {"3 × 3": 9, "4 × 4": 16}
_PIXMAP_CACHE_LIMIT = 512


class GridBalanceDialog(BaseDialog):
    """Rapid combination balancing; emits ``label_applied(index, label|None)``."""

    label_applied = Signal(int, object)

    # Input is ignored on a tile for this long after its image changes, and on
    # every tile after the combination switches by itself.
    SETTLE_S = 0.25
    ADVANCE_LOCK_S = 0.8

    def __init__(
        self,
        session: GridBalanceSession,
        image_paths: Sequence[Path | str],
        *,
        tile_edge: int = 170,
        parent=None,
    ) -> None:
        super().__init__(
            "Grid Balance — click every match", parent, QDialogButtonBox.Close
        )
        self.session = session
        self._paths = [str(p) for p in image_paths]
        self._tile_edge = int(tile_edge)
        self._pixmaps: "OrderedDict[str, QPixmap]" = OrderedDict()
        self._failed: Set[str] = set()
        self._tiles: List[_Tile] = []
        self._ready_at: List[float] = []
        self._tile_shortcuts: List[QShortcut] = []
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
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Header: what to click, how far along this combination is.
        head = QHBoxLayout()
        self.combo_label = QLabel()
        self.combo_label.setTextFormat(Qt.RichText)
        head.addWidget(self.combo_label, 1)
        self.position_label = QLabel()
        self.position_label.setStyleSheet("color: #aaa;")
        head.addWidget(self.position_label)
        layout.addLayout(head)

        prog = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFormat("%v / %m labeled")
        prog.addWidget(self.progress, 1)
        self.session_label = QLabel()
        self.session_label.setStyleSheet("color: #aaa;")
        prog.addWidget(self.session_label)
        layout.addLayout(prog)

        ref = QHBoxLayout()
        ref.setSpacing(6)
        ref_title = QLabel("Examples:")
        ref_title.setStyleSheet("color: #aaa;")
        ref.addWidget(ref_title)
        self._ref_tiles = []
        for _ in range(3):
            t = _Tile("", 64, root)
            t.setCursor(Qt.ArrowCursor)
            t.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            self._ref_tiles.append(t)
            ref.addWidget(t)
        self.ref_note = QLabel()
        self.ref_note.setStyleSheet("color: #888; font-size: 11px;")
        ref.addWidget(self.ref_note)
        ref.addStretch(1)
        layout.addLayout(ref)

        self.grid_host = QWidget(root)
        self.grid_layout = QGridLayout(self.grid_host)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(6)
        self.grid_host.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(self.grid_host, 0, Qt.AlignHCenter)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Grid:"))
        self.size_combo = QComboBox()
        self.size_combo.addItems(list(GRID_SIZES))
        self.size_combo.setCurrentIndex(
            list(GRID_SIZES.values()).index(session.grid_size)
            if session.grid_size in GRID_SIZES.values()
            else 0
        )
        self.size_combo.setFocusPolicy(Qt.NoFocus)
        self.size_combo.currentTextChanged.connect(self._on_size_changed)
        controls.addWidget(self.size_combo)
        controls.addStretch(1)
        self.undo_btn = self._button("↶  Undo  (Backspace)", self.undo)
        self.none_btn = self._button("⟳  None match  (Enter)", self.none_match)
        self.next_btn = self._button("⏭  Next label  (N)", self.next_label)
        self.none_btn.setStyleSheet("background: #0e639c; font-weight: bold;")
        for b in (self.undo_btn, self.none_btn, self.next_btn):
            controls.addWidget(b)
        layout.addLayout(controls)

        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self.status_label)
        self.add_content(root)

        for seq, slot in (
            ("Return", self.none_match),
            ("Enter", self.none_match),
            ("N", self.next_label),
            ("Backspace", self.undo),
            (QKeySequence.Undo, self.undo),
        ):
            self._shortcut(QKeySequence(seq), slot)

        self._build_grid()
        self._render()

    # ── actions ──────────────────────────────────────────────────────────
    def click_slot(self, slot: int) -> None:
        if not self._clickable(slot):
            return
        res = self.session.click(slot)
        if res is None:
            return
        self.label_applied.emit(res.index, res.label)
        self._tiles[slot].flash()
        self._settle([slot])
        if res.completed:
            done = res.label
            more = self.session.next_combo()
            # Hold input so a keypress meant for the old grid can't label the
            # new combination.
            self._settle(range(len(self._tiles)), self.ADVANCE_LOCK_S)
            self._set_status(
                f"✓ {done} balanced."
                + (
                    f" Now: {self.session.current}."
                    if more
                    else " All combinations are balanced."
                )
            )
        self._render()

    def none_match(self) -> None:
        if self.session.finished:
            return
        self.session.refresh()
        self._settle(range(len(self._tiles)))
        self._set_status("New images drawn.")
        self._render()

    def next_label(self) -> None:
        if self.session.finished:
            return
        prev = self.session.current
        more = self.session.next_combo()
        self._settle(range(len(self._tiles)))
        self._set_status(
            f"Skipped {prev}." if more else "No more combinations below target."
        )
        self._render()

    def undo(self) -> None:
        prev = self.session.current
        hit = self.session.undo()
        if hit is None:
            self._set_status("Nothing to undo.")
            return
        self.label_applied.emit(hit.index, None)
        changed = (
            range(len(self._tiles))
            if self.session.current != prev
            else [min(hit.slot, len(self._tiles) - 1)]
        )
        self._settle(changed)
        self._set_status(f"Undid {hit.combo} on image {hit.index}.")
        self._render()

    # ── view ─────────────────────────────────────────────────────────────
    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.setFocus()

    def _shortcut(self, seq, slot) -> QShortcut:
        sc = QShortcut(seq, self)
        sc.setAutoRepeat(False)  # holding a key must not label unseen images
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

    def _settle(self, slots, seconds: Optional[float] = None) -> None:
        until = time.monotonic() + (self.SETTLE_S if seconds is None else seconds)
        for s in slots:
            if 0 <= s < len(self._ready_at):
                self._ready_at[s] = max(self._ready_at[s], until)

    def _clickable(self, slot: int) -> bool:
        """Only a tile whose image is on screen, and has been for a moment."""
        if not 0 <= slot < len(self._tiles) or slot >= len(self.session.grid):
            return False
        if time.monotonic() < self._ready_at[slot]:
            return False
        idx = self.session.grid[slot]
        return idx is not None and self._paths[idx] in self._pixmaps

    def _keys(self) -> List[str]:
        return KEYS_4X4 if self.session.grid_size == 16 else KEYS_3X3

    def _build_grid(self) -> None:
        for t in self._tiles:
            t.setParent(None)
            t.deleteLater()
        for sc in self._tile_shortcuts:
            sc.setEnabled(False)
            sc.setParent(None)
            sc.deleteLater()
        self._tiles, self._tile_shortcuts = [], []
        k = self.session.grid_size
        self._ready_at = [0.0] * k
        cols = int(round(k**0.5))
        keys = self._keys()
        edge = self._tile_edge if k <= 9 else int(self._tile_edge * 0.8)
        for slot in range(k):
            key = keys[slot] if slot < len(keys) else ""
            tile = _Tile(key, edge, self.grid_host)
            tile.clicked.connect(lambda _=False, s=slot: self.click_slot(s))
            self.grid_layout.addWidget(tile, slot // cols, slot % cols)
            self._tiles.append(tile)
            if key:
                self._tile_shortcuts.append(
                    self._shortcut(QKeySequence(key), lambda s=slot: self.click_slot(s))
                )
        self._settle(range(k))
        self.grid_host.adjustSize()

    def _on_size_changed(self, text: str) -> None:
        size = GRID_SIZES.get(text)
        if size is None or size == self.session.grid_size:
            return
        self.session.resize(size)
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
            self._prefetcher.request([path], lambda p: _load_scaled(p, edge))
        return pm

    def _on_image_ready(self, path: str, image) -> None:
        while len(self._pixmaps) >= _PIXMAP_CACHE_LIMIT:
            self._pixmaps.popitem(last=False)
        pm = QPixmap.fromImage(image)
        pm.setDevicePixelRatio(self.devicePixelRatioF())
        self._pixmaps[path] = pm
        # A tile becomes clickable only once its image is visible.
        self._settle(
            [
                s
                for s, i in enumerate(self.session.grid)
                if i is not None and self._paths[i] == path
            ]
        )
        self._render_tiles()

    def _on_image_failed(self, path: str) -> None:
        """Unreadable image: swap it for another candidate (never labeled)."""
        self._failed.add(path)
        for idx, p in enumerate(self._paths):
            if p == path:
                self.session.exclude(idx)  # not drawn again for any combination
        self._replace_failed_tiles()
        n = len(self._failed)
        self._set_status(f"{n} unreadable image{'s' if n != 1 else ''} skipped.")
        self._render()

    def _replace_failed_tiles(self) -> None:
        for slot, idx in enumerate(list(self.session.grid)):
            while idx is not None and self._paths[idx] in self._failed:
                idx = self.session.replace(slot)
                self._settle([slot])

    def _render(self) -> None:
        s = self.session
        combo = s.current
        if combo is None:
            self.combo_label.setText(
                "<span style='font-size:18px; font-weight:bold;'>All done</span>"
            )
            self.position_label.setText("")
            self.progress.setRange(0, 1)
            self.progress.setValue(1)
            for b in (self.none_btn, self.next_btn):
                b.setEnabled(False)
        else:
            pretty = " · ".join(f"<b>{part}</b>" for part in s.parts(combo))
            self.combo_label.setText(
                "<span style='font-size:18px;'>Click every image that is "
                f"{pretty}</span>"
            )
            self.position_label.setText(
                f"Combination {s.pos + 1} of {len(s.combos)}  ·  "
                f"{s.remaining():,} candidates left"
            )
            self.progress.setRange(0, max(1, s.target))
            self.progress.setValue(min(s.counts[combo], s.target))
            for b in (self.none_btn, self.next_btn):
                b.setEnabled(True)
        self.undo_btn.setEnabled(bool(s.hits))
        self.session_label.setText(f"{s.session_labeled()} labeled this session")
        self._render_tiles()

    def _render_tiles(self) -> None:
        s = self.session
        self._replace_failed_tiles()
        for slot, tile in enumerate(self._tiles):
            idx = s.grid[slot] if slot < len(s.grid) else None
            tile.set_image(None if idx is None else self._pixmap(idx), idx is None)
        examples = s.examples() if s.current is not None else []
        for i, t in enumerate(self._ref_tiles):
            if i < len(examples):
                path = self._paths[examples[i]]
                failed = path in self._failed
                t.set_image(
                    None if failed else self._pixmap(examples[i]),
                    False,
                    "✕" if failed else "",
                )
                t.show()
            else:
                t.hide()
        self.ref_note.setText(
            "" if examples or s.current is None else "no labeled examples yet"
        )

    def done(self, result: int) -> None:  # noqa: D401 - Qt override
        self._prefetcher.shutdown()
        super().done(result)
