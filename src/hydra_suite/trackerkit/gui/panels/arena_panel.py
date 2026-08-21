"""The arena bar: an arena-centric replacement for the old ROI toolbar.

Two states. Empty shows only a sentence and two add buttons. Editing shows
arena navigation, the zone tools, and the overlap warning. The
include/exclude combo is gone -- zone role is chosen by pressing "+ Circle"
versus "- Circle", so the user never sets a mode before drawing.

One arena is exactly a plain ROI: with a single arena, ``n_arenas == 1``
suppresses the ``arena_id`` column and every per-arena path degenerates, so
a user who only wants to mask out junk never meets arena numbering.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.trackerkit.arena_geometry import overlapping_arena_pairs


class ArenaPanel(QWidget):
    """Arena navigation, zone tools and the overlap lock."""

    arena_changed = Signal(int)
    add_single_requested = Signal()
    add_grid_requested = Signal()
    clear_arena_requested = Signal(int)
    draw_requested = Signal(str, str)
    finish_requested = Signal()
    undo_requested = Signal()
    clear_all_requested = Signal()
    crop_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._shapes: list[dict[str, Any]] = []
        self._frame_size = (0, 0)
        self._current = 0
        self._pending_new = False
        self._shape_valid = False

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(4)

        self.stack = QStackedWidget()
        self.empty_widget = self._build_empty()
        self.editing_widget = self._build_editing()
        self.stack.addWidget(self.empty_widget)
        self.stack.addWidget(self.editing_widget)
        root.addWidget(self.stack)

        self.lbl_warning = QLabel("")
        self.lbl_warning.setWordWrap(True)
        self.lbl_warning.setStyleSheet(
            "color: #ffffff; font-weight: bold; padding: 6px; "
            "background-color: #8a1f1f; border-radius: 4px;"
        )
        self.lbl_warning.setVisible(False)
        root.addWidget(self.lbl_warning)

        self.refresh()

    def _build_empty(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_default = QLabel("By default, the whole video is used.")
        self.lbl_default.setStyleSheet("color: #cccccc;")
        self.btn_add_single = QPushButton("+ Add Single Arena")
        self.btn_add_grid = QPushButton("+ Add Grid of Arenas")
        self.btn_add_single.clicked.connect(self.add_single_requested.emit)
        self.btn_add_grid.clicked.connect(self.add_grid_requested.emit)
        layout.addWidget(self.lbl_default)
        layout.addWidget(self.btn_add_single)
        layout.addWidget(self.btn_add_grid)
        layout.addStretch()
        return widget

    def _build_editing(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_current = QLabel("Currently labelling: Arena 1")
        self.lbl_current.setStyleSheet("font-weight: bold; color: #cccccc;")
        self.btn_prev = QPushButton("< Previous")
        self.btn_next = QPushButton("Next >")
        self.btn_add_new = QPushButton("+ Add new arena")
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next.clicked.connect(lambda: self._step(1))
        self.btn_add_new.clicked.connect(self.begin_new_arena)

        self.lbl_hint = QLabel(
            "Add Inclusion and Exclusion Zones "
            "(Left-click marks, Right-click removes the last point)"
        )
        self.lbl_hint.setStyleSheet("color: #4fc1ff; font-size: 11px;")

        self.btn_clear_arena = QPushButton("Clear Arena")
        self.btn_clear_arena.clicked.connect(
            lambda: self.clear_arena_requested.emit(self._current)
        )
        self.btn_add_circle = QPushButton("+ Circle")
        self.btn_sub_circle = QPushButton("- Circle")
        self.btn_add_polygon = QPushButton("+ Polygon")
        self.btn_sub_polygon = QPushButton("- Polygon")
        self.btn_add_circle.clicked.connect(
            lambda: self.draw_requested.emit("circle", "include")
        )
        self.btn_sub_circle.clicked.connect(
            lambda: self.draw_requested.emit("circle", "exclude")
        )
        self.btn_add_polygon.clicked.connect(
            lambda: self.draw_requested.emit("polygon", "include")
        )
        self.btn_sub_polygon.clicked.connect(
            lambda: self.draw_requested.emit("polygon", "exclude")
        )
        self.btn_finish = QPushButton("Finish Shape")
        self.btn_finish.setEnabled(False)
        self.btn_finish.clicked.connect(self.finish_requested.emit)
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.clicked.connect(self.undo_requested.emit)

        self.btn_overflow = QToolButton()
        self.btn_overflow.setText("...")
        self.btn_overflow.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.btn_overflow)
        menu.addAction("Add Grid of Arenas", self.add_grid_requested.emit)
        menu.addAction("Clear All", self.clear_all_requested.emit)
        menu.addAction("Crop Video to ROI", self.crop_requested.emit)
        self.btn_overflow.setMenu(menu)

        for w in (
            self.lbl_current,
            self.btn_prev,
            self.btn_next,
            self.btn_add_new,
            self._separator(),
            self.lbl_hint,
            self.btn_clear_arena,
            self.btn_add_circle,
            self.btn_sub_circle,
            self.btn_add_polygon,
            self.btn_sub_polygon,
            self.btn_finish,
            self.btn_undo,
            self.btn_overflow,
        ):
            layout.addWidget(w)
        layout.addStretch()
        return widget

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setStyleSheet("color: #3e3e42;")
        return line

    # -- state ------------------------------------------------------------

    @property
    def current_arena(self) -> int:
        return self._current

    def arena_ids(self) -> list[int]:
        return sorted(
            {
                int(s.get("arena_id", 0))
                for s in self._shapes
                if s.get("mode", "include") == "include"
            }
        )

    def set_shapes(self, shapes: list[dict[str, Any]] | None) -> None:
        self._shapes = list(shapes or [])
        if self._shapes:
            self._pending_new = False
        self.refresh()

    def set_frame_size(self, width: int, height: int) -> None:
        self._frame_size = (int(width), int(height))
        self.refresh()

    def set_current_arena(self, arena_id: int) -> None:
        self._current = int(arena_id)
        self.refresh()

    def set_shape_valid(self, valid: bool) -> None:
        self._shape_valid = bool(valid)
        self.btn_finish.setEnabled(self._shape_valid)

    def begin_new_arena(self) -> int:
        """Start a fresh arena; it holds no shapes until the user draws one."""
        from hydra_suite.trackerkit.engine_params import next_free_arena_id

        self._current = next_free_arena_id(self._shapes)
        self._pending_new = True
        self.refresh()
        self.arena_changed.emit(self._current)
        return self._current

    def shapes_after_clearing(self, arena_id: int) -> list[dict[str, Any]]:
        """The shape list with *arena_id*'s shapes removed; the arena remains.

        Numbering is untouched -- arena ids appear in the exported
        ``arena_id`` column, so renumbering would silently change what a
        number refers to.
        """
        return [s for s in self._shapes if int(s.get("arena_id", 0)) != int(arena_id)]

    def _step(self, delta: int) -> None:
        ids = self.arena_ids()
        if self._current not in ids:
            return
        index = ids.index(self._current) + delta
        if 0 <= index < len(ids):
            self._current = ids[index]
            self.refresh()
            self.arena_changed.emit(self._current)

    # -- lock -------------------------------------------------------------

    def blocking_pairs(self) -> list[tuple[int, int]]:
        width, height = self._frame_size
        if not width or not height:
            return []
        return overlapping_arena_pairs(self._shapes, width, height)

    def can_track(self) -> tuple[bool, str]:
        """Whether tracking may start, and why not if it may not."""
        pairs = self.blocking_pairs()
        if not pairs:
            return (True, "")
        listed = ", ".join(f"Arena {a + 1} and Arena {b + 1}" for a, b in pairs)
        return (
            False,
            f"Arenas overlap: {listed}. Each animal must belong to exactly one "
            "arena, so tracking cannot start until the overlaps are resolved.",
        )

    def refresh(self) -> None:
        ids = self.arena_ids()
        if not self._shapes and not self._pending_new:
            self.stack.setCurrentWidget(self.empty_widget)
            self.lbl_warning.setVisible(False)
            return
        self.stack.setCurrentWidget(self.editing_widget)
        self.lbl_current.setText(f"Currently labelling: Arena {self._current + 1}")

        pairs = self.blocking_pairs()
        conflicts = sorted(
            {
                (b if a == self._current else a)
                for a, b in pairs
                if self._current in (a, b)
            }
        )
        current_blocked = bool(conflicts)
        current_empty = self._current not in ids

        if current_blocked:
            listed = ", ".join(f"Arena {c + 1}" for c in conflicts)
            self.lbl_warning.setText(
                f"Arena {self._current + 1} overlaps {listed}. "
                "Resolve the overlap before moving on -- an animal in the shared "
                "region cannot be assigned to a single arena."
            )
        elif pairs:
            self.lbl_warning.setText(self.can_track()[1])
        self.lbl_warning.setVisible(bool(pairs))

        index = ids.index(self._current) if self._current in ids else -1
        self.btn_prev.setEnabled(not current_blocked and index > 0)
        self.btn_next.setEnabled(
            not current_blocked and index >= 0 and index < len(ids) - 1
        )
        self.btn_add_new.setEnabled(not current_blocked and not current_empty)
        self.btn_clear_arena.setEnabled(not current_empty)
        self.btn_undo.setEnabled(bool(self._shapes))
