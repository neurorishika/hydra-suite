"""The compact (paired-rows) layout of the shared SAHI settings widget (S7).

``SliceWidgetCapabilities(layout="compact")`` puts two label/field pairs on
one grid row (``COMPACT_PAIRS``), folds the derived notes into one summary
line under the grid, and keeps a host's advanced note under the Advanced
rows. Split out of ``slice_settings`` (size guideline); each function takes
the widget ``w``. Shared layer: imports only Qt (never an app layer).
"""

from __future__ import annotations

import re

from PySide6.QtWidgets import QLabel, QWidget

from .slice_settings_controls import FULL_WIDTH, row_specs
from .slice_settings_parts import COMPACT_LABELS, COMPACT_PAIRS, hbox

# Columns: label | field | gap | label | field. The empty gap column keeps
# the right-hand pair clear of the left field's source badge.
_SPAN = 5
_GAP_PX = 6

Cell = tuple[QWidget, int, int, int]  # (widget, row, column, column span)


def build_compact_grid(w, controls: QWidget) -> None:
    """Create the labels and the summary row, then place the shown cells.

    Cells are (re)placed by :func:`pack_compact` whenever the set of shown
    rows changes, so a pair whose partner is hidden moves to the left edge
    instead of leaving a hole.
    """
    w._grid.setColumnStretch(_SPAN - 1, 1)
    w._grid.setColumnMinimumWidth(2, _GAP_PX)
    w._grid_items = {}
    for key, text, control, _badge in row_specs(w):
        label = None
        widgets: list[QWidget] = [control]
        if text is not None and key not in FULL_WIDTH and key != "advanced":
            label = QLabel(COMPACT_LABELS.get(key, text))
            label.setToolTip(w._label_tooltip(control))
            label.setParent(controls)
            w._rows[key] = (label, control)
            widgets.append(label)
        # Parented now: a cell shown later must never become a top-level.
        control.setParent(controls)
        w._grid_items[key] = (label, control)
        w._row_widgets[key] = widgets
    w._summary_row = hbox(
        w.lbl_slice_summary,
        w.lbl_slice_overlap_minimum,
        w.btn_slice_overlap_raise,
    )
    w._summary_row.layout().setSpacing(4)
    w._summary_row.setParent(controls)
    pack_compact(w)


def compact_plan(w) -> list[Cell]:
    """Every shown compact cell, in reading order."""
    partners = dict(COMPACT_PAIRS)
    items = w._grid_items

    def shown(key: str) -> bool:
        return key in items and not items[key][1].isHidden()

    plan: list[Cell] = []
    placed: set[str] = set()
    row = 0
    for key, (label, control) in items.items():
        if key in placed or not shown(key):
            continue
        placed.add(key)
        if label is None:
            plan.append((control, row, 0, _SPAN))
            row += 1
            continue
        mate = partners.get(key)
        plan.append((label, row, 0, 1))
        if mate and shown(mate) and items[mate][0] is not None:
            placed.add(mate)
            mate_label, mate_control = items[mate]
            plan += [
                (control, row, 1, 1),
                (mate_label, row, 3, 1),
                (mate_control, row, 4, 1),
            ]
        else:
            plan.append((control, row, 1, _SPAN - 1))
        row += 1
        summary_placed = any(cell[0] is w._summary_row for cell in plan)
        if "overlap" in placed and not w._summary_row.isHidden() and not summary_placed:
            plan.append((w._summary_row, row, 0, _SPAN))
            row += 1
    if w._advanced_note is not None:
        plan.append((w._advanced_note, row, 0, _SPAN))
    return plan


def pack_compact(w) -> None:
    """Re-place the grid cells, only when the shown set changed."""
    plan = compact_plan(w)
    signature = tuple((id(cell[0]),) + cell[1:] for cell in plan)
    if signature == w._packed:
        return
    grid = w._grid
    for widget, *_cell in w._packed_plan:
        grid.removeWidget(widget)
    for widget, row, column, span in plan:
        grid.addWidget(widget, row, column, 1, span)
    w._packed = signature
    w._packed_plan = plan
    grid.invalidate()


def refresh_summary(w, mode: str) -> None:
    """The derived notes as ONE muted line under the grid.

    The overlap note keeps its own label (and colour) at the end of the
    line; a below-minimum warning never elides -- the muted part gives way
    first. The full text is always in the tooltip.
    """
    tile_full = w.lbl_slice_tile_size.text()
    # "(48 px ÷ 0.1)" is the derivation, not the result: tooltip only.
    parts = [re.sub(r" \([^()]*÷[^()]*\)$", "", tile_full)]
    full = [tile_full]
    if w._role == "infer_yolo" and mode == "auto_object":
        imgsz = w._model_input_size
        px = float(w.spin_slice_object_fraction.value()) * imgsz
        parts.append(f"≈{px:.0f} px at {imgsz}")
        full.append(w.lbl_slice_scale_px.text())
    overlap = w.lbl_slice_overlap_minimum
    if overlap.text():
        full.append(overlap.text())
    w.lbl_slice_summary.setText(" · ".join(parts) + (" ·" if overlap.text() else ""))
    w.lbl_slice_summary.setToolTip(
        " · ".join(full)
        + "\n\nTile size, the object's size at the model input, and the "
        "whole-animal overlap minimum, derived from the settings above."
    )
    warning = not w.btn_slice_overlap_raise.isHidden()
    overlap.setMinimumWidth(
        overlap.fontMetrics().horizontalAdvance(overlap.text()) + 4 if warning else 0
    )
