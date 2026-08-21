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
``ArenaGridDialog`` is a thin Qt wrapper around it with a live preview drawn
through ``ArenaCanvas.render_overlay`` -- the same renderer the main window
preview uses, so the two previews cannot visually drift apart.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.trackerkit.arena_geometry import (
    generate_grid_shapes,
    max_grid_extent,
    min_pitch,
)
from hydra_suite.widgets.dialogs import BaseDialog


class ArenaGridDialog(BaseDialog):
    """Bulk-entry dialog: rows x cols, origin, pitch, shape, size and rotation.

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

        if self._reference_frame is not None:
            avg_dim = (
                self._reference_frame.width() + self._reference_frame.height()
            ) / 2.0
            default_radius = max(1, round(0.20 * avg_dim))
        else:
            default_radius = 20  # no reference frame available -- keep the old fallback

        form_group = QGroupBox("Grid layout")
        form = QFormLayout(form_group)

        self.combo_shape_type = QComboBox()
        self.combo_shape_type.addItems(["Circle", "Rectangle"])
        form.addRow("Shape:", self.combo_shape_type)

        self.spin_radius = QSpinBox()
        self.spin_radius.setRange(1, 100000)
        self.spin_radius.setValue(default_radius)
        self.row_radius = QLabel("Radius:")
        radius_slider_max = (
            max(1, round(avg_dim)) if self._reference_frame is not None else 2000
        )
        self.slider_radius = self._pair_with_slider(
            form, self.row_radius, self.spin_radius, 1, radius_slider_max
        )

        self.spin_width = QSpinBox()
        self.spin_width.setRange(1, 100000)
        self.spin_width.setValue(default_radius * 2)
        self.row_width = QLabel("Width:")
        self.slider_width = self._pair_with_slider(
            form, self.row_width, self.spin_width, 1, radius_slider_max
        )

        self.spin_height = QSpinBox()
        self.spin_height.setRange(1, 100000)
        self.spin_height.setValue(default_radius * 2)
        self.row_height = QLabel("Height:")
        self.slider_height = self._pair_with_slider(
            form, self.row_height, self.spin_height, 1, radius_slider_max
        )

        self.spin_origin_x = QSpinBox()
        self.spin_origin_x.setRange(0, 100000)
        self.spin_origin_x.setValue(default_radius)
        origin_x_slider_max = (
            self._reference_frame.width() if self._reference_frame is not None else 2000
        )
        self.slider_origin_x = self._pair_with_slider(
            form, "Arena 1 Centre X:", self.spin_origin_x, 0, origin_x_slider_max
        )

        self.spin_origin_y = QSpinBox()
        self.spin_origin_y.setRange(0, 100000)
        self.spin_origin_y.setValue(default_radius)
        origin_y_slider_max = (
            self._reference_frame.height()
            if self._reference_frame is not None
            else 2000
        )
        self.slider_origin_y = self._pair_with_slider(
            form, "Arena 1 Centre Y:", self.spin_origin_y, 0, origin_y_slider_max
        )

        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(1, 100)
        self.spin_rows.setValue(1)
        self.slider_rows = self._pair_with_slider(
            form, "Rows:", self.spin_rows, 1, self.spin_rows.maximum()
        )

        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(1, 100)
        self.spin_cols.setValue(1)
        self.slider_cols = self._pair_with_slider(
            form, "Columns:", self.spin_cols, 1, self.spin_cols.maximum()
        )

        self.spin_pitch_x = QSpinBox()
        self.spin_pitch_x.setRange(1, 100000)
        self.row_pitch_x = QLabel("X spacing:")
        pitch_x_slider_max = (
            self._reference_frame.width() if self._reference_frame is not None else 2000
        )
        self.slider_pitch_x = self._pair_with_slider(
            form,
            self.row_pitch_x,
            self.spin_pitch_x,
            self.spin_pitch_x.minimum(),
            pitch_x_slider_max,
        )

        self.spin_pitch_y = QSpinBox()
        self.spin_pitch_y.setRange(1, 100000)
        self.row_pitch_y = QLabel("Y spacing:")
        pitch_y_slider_max = (
            self._reference_frame.height()
            if self._reference_frame is not None
            else 2000
        )
        self.slider_pitch_y = self._pair_with_slider(
            form,
            self.row_pitch_y,
            self.spin_pitch_y,
            self.spin_pitch_y.minimum(),
            pitch_y_slider_max,
        )

        rotation_row = QWidget()
        rotation_layout = QHBoxLayout(rotation_row)
        rotation_layout.setContentsMargins(0, 0, 0, 0)
        self.slider_rotation = QSlider(Qt.Horizontal)
        # Half-degree ticks: the slider is integer-valued, so it counts halves.
        self.slider_rotation.setRange(-90, 90)
        self.spin_rotation = QDoubleSpinBox()
        self.spin_rotation.setRange(-45.0, 45.0)
        self.spin_rotation.setSingleStep(0.5)
        self.spin_rotation.setDecimals(1)
        self.spin_rotation.setSuffix(" deg")
        rotation_layout.addWidget(self.slider_rotation)
        rotation_layout.addWidget(self.spin_rotation)
        form.addRow("Rotation (about arena 1):", rotation_row)

        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet("color: #6a6a6a;")
        form.addRow("", self.lbl_summary)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(320, 240)
        self.preview_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
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

        self.combo_shape_type.currentTextChanged.connect(self._on_shape_changed)
        for spin in (self.spin_radius, self.spin_width, self.spin_height):
            spin.valueChanged.connect(self._sync_pitch_floors)
        for spin in (self.spin_rows, self.spin_cols):
            spin.valueChanged.connect(self._sync_spacing_visibility)
        for spin in (self.spin_origin_x, self.spin_origin_y):
            spin.valueChanged.connect(self._sync_extent_caps)
        for spin in (self.spin_pitch_x, self.spin_pitch_y):
            spin.valueChanged.connect(self._sync_extent_caps)
        self.slider_rotation.valueChanged.connect(self._on_slider_rotation)
        self.spin_rotation.valueChanged.connect(self._on_spin_rotation)
        for widget in (
            self.spin_rows,
            self.spin_cols,
            self.spin_origin_x,
            self.spin_origin_y,
            self.spin_pitch_x,
            self.spin_pitch_y,
            self.spin_radius,
            self.spin_width,
            self.spin_height,
        ):
            widget.valueChanged.connect(self._update_preview)
        self.combo_shape_type.currentTextChanged.connect(self._update_preview)

        self._on_shape_changed()
        self._update_preview()

    def _pair_with_slider(
        self,
        form: QFormLayout,
        label: str | QLabel,
        spin: QSpinBox,
        slider_min: int,
        slider_max: int,
    ) -> QSlider:
        """Add *spin* to *form* as one row, paired with a QSlider covering the
        same integer range. Both stay in sync; typing a value outside the
        slider's range still works in the spinbox (the slider just clamps to
        its own end), since the spinbox remains the authoritative value.
        """
        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(slider_min), max(int(slider_min) + 1, int(slider_max)))
        slider.setValue(max(slider_min, min(slider_max, spin.value())))

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(slider)
        row_layout.addWidget(spin)
        form.addRow(label, row)

        slider.valueChanged.connect(spin.setValue)

        def _on_spin_changed(value: int, slider=slider) -> None:
            # blockSignals so a spin value beyond the slider's range doesn't
            # bounce back through slider.valueChanged -> spin.setValue and
            # clamp the spinbox itself -- the spinbox is meant to stay
            # authoritative (same pattern Rotation already uses to avoid a
            # slider<->spin feedback loop).
            slider.blockSignals(True)
            slider.setValue(max(slider.minimum(), min(slider.maximum(), value)))
            slider.blockSignals(False)

        spin.valueChanged.connect(_on_spin_changed)
        return slider

    def _shape_key(self) -> str:
        """Internal shape id for the geometry helpers."""
        return (
            "circle" if self.combo_shape_type.currentText() == "Circle" else "polygon"
        )

    def _size_pair(self) -> tuple[int, int]:
        """(size_x, size_y): diameter/diameter for circles, width/height for rects."""
        if self._shape_key() == "circle":
            diameter = self.spin_radius.value() * 2
            return (diameter, diameter)
        return (self.spin_width.value(), self.spin_height.value())

    def _on_shape_changed(self, *_args) -> None:
        is_circle = self._shape_key() == "circle"
        for widget in (self.row_radius, self.spin_radius, self.slider_radius):
            widget.setVisible(is_circle)
        for widget in (
            self.row_width,
            self.spin_width,
            self.slider_width,
            self.row_height,
            self.spin_height,
            self.slider_height,
        ):
            widget.setVisible(not is_circle)
        self._sync_pitch_floors()

    def _sync_pitch_floors(self, *_args) -> None:
        """Clamp spacing to the tightest value that cannot overlap.

        Flooring here means the generator can never emit a layout the overlap
        lock would immediately reject.
        """
        size_x, size_y = self._size_pair()
        floor_x, floor_y = min_pitch(self._shape_key(), size_x, size_y=size_y)
        for spin, slider, floor in (
            (self.spin_pitch_x, self.slider_pitch_x, floor_x),
            (self.spin_pitch_y, self.slider_pitch_y, floor_y),
        ):
            was_at_floor = spin.value() <= spin.minimum()
            spin.setMinimum(int(floor))
            # blockSignals: QSlider.setMinimum can raise both the maximum and
            # the current value when the new floor exceeds the slider's
            # current maximum, firing valueChanged -> spin.setValue and
            # clobbering a larger user-typed spinbox value -- same
            # feedback-loop class _pair_with_slider already guards against.
            slider.blockSignals(True)
            slider.setMinimum(int(floor))
            slider.blockSignals(False)
            if was_at_floor or spin.value() < floor:
                spin.setValue(int(floor))
        self._sync_spacing_visibility()
        self._sync_extent_caps()

    def _sync_spacing_visibility(self, *_args) -> None:
        """Spacing is meaningless with one row/column, so it stays hidden."""
        multi_col = self.spin_cols.value() > 1
        multi_row = self.spin_rows.value() > 1
        self.row_pitch_x.setVisible(multi_col)
        self.spin_pitch_x.setVisible(multi_col)
        self.slider_pitch_x.setVisible(multi_col)
        self.row_pitch_y.setVisible(multi_row)
        self.spin_pitch_y.setVisible(multi_row)
        self.slider_pitch_y.setVisible(multi_row)

    def _sync_extent_caps(self, *_args) -> None:
        """Cap rows/cols so every arena CENTRE stays inside the frame."""
        if self._reference_frame is None:
            return
        origin_x, origin_y = self.spin_origin_x.value(), self.spin_origin_y.value()
        max_rows, max_cols = max_grid_extent(
            origin_x,
            origin_y,
            self.spin_pitch_x.value(),
            self.spin_pitch_y.value(),
            self._reference_frame.width(),
            self._reference_frame.height(),
            rotation_deg=self.spin_rotation.value(),
        )
        self.spin_rows.setMaximum(max_rows)
        self.spin_cols.setMaximum(max_cols)
        self.slider_rows.setMaximum(max_rows)
        self.slider_cols.setMaximum(max_cols)

    def _on_slider_rotation(self, ticks: int) -> None:
        self.spin_rotation.setValue(ticks / 2.0)

    def _on_spin_rotation(self, degrees: float) -> None:
        self.slider_rotation.blockSignals(True)
        self.slider_rotation.setValue(int(round(degrees * 2)))
        self.slider_rotation.blockSignals(False)
        self._sync_extent_caps()
        self._update_preview()

    def _current_shapes(self) -> list[dict[str, Any]]:
        """The grid shapes for the dialog's current widget values."""
        size_x, size_y = self._size_pair()
        origin_x, origin_y = self.spin_origin_x.value(), self.spin_origin_y.value()
        return generate_grid_shapes(
            self.spin_rows.value(),
            self.spin_cols.value(),
            origin_x,
            origin_y,
            self.spin_pitch_x.value(),
            self.spin_pitch_y.value(),
            size_x,
            shape_type=self._shape_key(),
            first_arena_id=self._first_arena_id,
            size_y=size_y,
            rotation_deg=self.spin_rotation.value(),
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

        from hydra_suite.trackerkit.gui.widgets.arena_canvas import ArenaCanvas

        renderer = ArenaCanvas()
        renderer.set_frame(image)
        renderer.set_shapes(shapes)
        target_w = self.preview_label.width() or 320
        target_h = self.preview_label.height() or 240
        zoom_w = target_w / max(1, image.width())
        zoom_h = target_h / max(1, image.height())
        renderer.set_zoom(min(1.0, zoom_w, zoom_h))
        pixmap = QPixmap(renderer.width(), renderer.height())
        pixmap.fill(Qt.black)
        painter = QPainter(pixmap)
        painter.drawPixmap(0, 0, renderer._scaled)
        renderer.render_overlay(painter)
        painter.end()
        self.preview_label.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_preview()

    def accepted_shapes(self) -> list[dict[str, Any]]:
        """The generated shapes for the dialog's current widget values.

        Callers should read this after ``exec()`` returns ``QDialog.Accepted``
        and extend their own ``roi_shapes`` list with the result.
        """
        return self._current_shapes()
