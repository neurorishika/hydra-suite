"""Bulk arena entry: lay down an R x C grid of arena shapes.

Hand-drawing 96 wells is impractical, so this generates the shapes in one go.
The output is ordinary ``roi_shapes`` entries with sequential ``arena_id``s
that stay individually editable afterwards -- this is a bulk-entry
convenience, not a separate mode. Nothing downstream distinguishes a
generated shape from a hand-drawn one: no new shape type, no marker field,
no separate storage.

``generate_grid_shapes`` is a pure function with no Qt dependency so it can
be tested without a display (see ``project_main_suite_blockers`` -- some GUI
tests crash the interpreter, so geometry logic must be testable standalone).
``ArenaGridDialog`` is a thin Qt wrapper around it with a live preview.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.trackerkit.arena_geometry import generate_grid_shapes
from hydra_suite.widgets.dialogs import BaseDialog


class ArenaGridDialog(BaseDialog):
    """Bulk-entry dialog: rows x cols, origin, pitch, shape and size.

    A thin wrapper over :func:`generate_grid_shapes` -- all geometry lives
    in that pure function; this class only collects spin-box values, shows a
    live preview drawn over the current reference frame, and hands back the
    generated shapes via :meth:`accepted_shapes`.

    ``first_arena_id`` must be the next free arena id across ALL existing
    shapes (hand-drawn or previously generated) so a caller can add a grid
    alongside arenas that already exist without id collisions -- computing
    that value is the caller's job (e.g.
    ``max((s["arena_id"] for s in roi_shapes), default=-1) + 1``).
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        reference_frame=None,
        first_arena_id: int = 0,
    ) -> None:
        super().__init__("Generate Arena Grid", parent)
        self._reference_frame = reference_frame
        self._first_arena_id = int(first_arena_id)

        form_group = QGroupBox("Grid layout")
        form = QFormLayout(form_group)

        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(1, 100)
        self.spin_rows.setValue(8)
        form.addRow("Rows:", self.spin_rows)

        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(1, 100)
        self.spin_cols.setValue(12)
        form.addRow("Columns:", self.spin_cols)

        self.spin_origin_x = QSpinBox()
        self.spin_origin_x.setRange(0, 100000)
        self.spin_origin_x.setValue(50)
        form.addRow("Origin X (centre of first arena):", self.spin_origin_x)

        self.spin_origin_y = QSpinBox()
        self.spin_origin_y.setRange(0, 100000)
        self.spin_origin_y.setValue(50)
        form.addRow("Origin Y (centre of first arena):", self.spin_origin_y)

        self.spin_pitch_x = QSpinBox()
        self.spin_pitch_x.setRange(1, 100000)
        self.spin_pitch_x.setValue(100)
        form.addRow("Pitch X (centre-to-centre):", self.spin_pitch_x)

        self.spin_pitch_y = QSpinBox()
        self.spin_pitch_y.setRange(1, 100000)
        self.spin_pitch_y.setValue(100)
        form.addRow("Pitch Y (centre-to-centre):", self.spin_pitch_y)

        self.spin_size = QSpinBox()
        self.spin_size.setRange(1, 100000)
        self.spin_size.setValue(40)
        form.addRow("Size (diameter / edge length):", self.spin_size)

        self.combo_shape_type = QComboBox()
        self.combo_shape_type.addItems(["circle", "polygon"])
        form.addRow("Shape:", self.combo_shape_type)

        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet("color: #6a6a6a;")
        form.addRow("", self.lbl_summary)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(320, 240)
        self.preview_label.setStyleSheet(
            "background-color: #000000; border: 1px solid #3e3e42;"
        )

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(form_group)
        preview_col = QVBoxLayout()
        preview_col.addWidget(QLabel("Preview"))
        preview_col.addWidget(self.preview_label)
        layout.addLayout(preview_col)

        self.add_content(container)

        for widget in (
            self.spin_rows,
            self.spin_cols,
            self.spin_origin_x,
            self.spin_origin_y,
            self.spin_pitch_x,
            self.spin_pitch_y,
            self.spin_size,
        ):
            widget.valueChanged.connect(self._update_preview)
        self.combo_shape_type.currentTextChanged.connect(self._update_preview)

        self._update_preview()

    def _current_shapes(self) -> list[dict[str, Any]]:
        """Compute the grid shapes for the dialog's current widget values."""
        return generate_grid_shapes(
            self.spin_rows.value(),
            self.spin_cols.value(),
            self.spin_origin_x.value(),
            self.spin_origin_y.value(),
            self.spin_pitch_x.value(),
            self.spin_pitch_y.value(),
            self.spin_size.value(),
            shape_type=self.combo_shape_type.currentText(),
            first_arena_id=self._first_arena_id,
        )

    def _update_preview(self, *_args) -> None:
        shapes = self._current_shapes()
        self.lbl_summary.setText(
            f"{len(shapes)} arena(s), ids {self._first_arena_id}.."
            f"{self._first_arena_id + len(shapes) - 1}"
        )

        if self._reference_frame is not None:
            image = self._reference_frame.copy()
        else:
            image = None

        if image is None:
            self.preview_label.setText("(no reference frame)")
            return

        pixmap = QPixmap.fromImage(image)
        painter = QPainter(pixmap)
        painter.setPen(QPen(Qt.cyan, 2))
        for shape in shapes:
            if shape["type"] == "circle":
                cx, cy, radius = shape["params"]
                painter.drawEllipse(
                    int(cx - radius),
                    int(cy - radius),
                    int(2 * radius),
                    int(2 * radius),
                )
            else:
                points = [QPoint(int(x), int(y)) for x, y in shape["params"]]
                painter.drawPolygon(points)
        painter.end()

        scaled = pixmap.scaled(
            self.preview_label.width() or 320,
            self.preview_label.height() or 240,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def accepted_shapes(self) -> list[dict[str, Any]]:
        """The generated shapes for the dialog's current widget values.

        Callers should read this after ``exec()`` returns ``QDialog.Accepted``
        and extend their own ``roi_shapes`` list with the result.
        """
        return self._current_shapes()
