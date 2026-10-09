"""Vocabulary, ranges and small builders for the shared SAHI settings widget.

Split out of ``slice_settings`` to keep the widget module readable. Shared
layer: imports only Qt and ``utils`` (never an app layer).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

from hydra_suite.utils.slice_geometry import tile_size_for_mode
from hydra_suite.utils.tiling_spec import (
    FRACTION_MAX,
    FRACTION_MIN,
    OVERLAP_MARGIN,
    OVERLAP_MAX,
)

ROLES = ("infer_yolo", "train_yolo", "train_sam3", "escalate_sam3", "escalate_sam2")

# Which defaults row of utils.tiling_spec.BACKEND_DEFAULTS each role reads.
ROLE_BACKEND = {
    "infer_yolo": "yolo_infer",
    "train_yolo": "yolo_train",
    "train_sam3": "sam3",
    "escalate_sam3": "sam3",
    "escalate_sam2": "sam2",
}
TRAIN_ROLES = ("train_yolo", "train_sam3")
ESCALATE_ROLES = ("escalate_sam3", "escalate_sam2")

# Ranges. Slice size and fractions mirror TilingSpec / tile_size_for_mode.
SLICE_SIZE_MAX = 8192
BODY_PX_MAX = 16384.0
# Sam3LoraParams.tile_overlap's own contract is [0, 1) (decision 27): a saved
# 0.95 must not be clamped to the shared 0.9 ceiling on load.
SAM3_TRAIN_OVERLAP_MAX = 0.99
# SAM2 owner tiles overlap by core.inference.semantic.tiling.DEFAULT_OVERLAP.
# The host passes the real constant via SliceWidgetCapabilities.fixed_overlap;
# this mirror only seeds a widget built without one.
SAM2_FIXED_OVERLAP_FALLBACK = 0.5
SEAM_MARGIN_MAX_PX = 64

# (enum, label, tooltip) -- labels are TrackerKit's vocabulary; the enum is
# what every reader stores (currentData/findData, never the label text).
GEOMETRY_ITEMS = (
    (
        "auto_model",
        "Use model input size",
        "Tiles are the model input size (auto_model).",
    ),
    (
        "auto_object",
        "Fit to animal size",
        "Tiles are sized so one animal fills the object scale: tile = body "
        "size / object scale (auto_object).",
    ),
    (
        "custom",
        "Custom tile size",
        "Tiles use the width and height entered below (custom).",
    ),
)
MERGE_POLICY_ITEMS = (
    (
        "greedy_nmm",
        "Greedy NMM",
        "Merge overlapping duplicates into one detection (greedy_nmm). A saved "
        "'nmm' runs the same code path and is shown here.",
    ),
    ("nms", "NMS", "Keep the highest-scoring duplicate and drop the rest (nms)."),
)
MERGE_METRIC_ITEMS = (
    ("ios", "IoS", "Intersection over the smaller box (ios)."),
    ("iou", "IoU", "Intersection over union (iou)."),
    ("polygon_iou", "Polygon IoU", "Mask polygon intersection over union."),
)
FRAGMENT_ITEMS = (
    ("drop", "Drop fragment", "Below the floor the fragment's label is dropped."),
    (
        "crowd",
        "Mark is_crowd",
        "Below the floor the fragment is kept but marked is_crowd.",
    ),
    (
        "mask",
        "Mask fragment",
        "Below the floor the label is dropped and its pixels painted out.",
    ),
)

SOURCE_DESCRIPTIONS = {
    "user": "entered by you",
    "override": "your override of a derived value",
    "profile": "the selected calibration profile",
    "stamped": "stamped on the model",
    "project": "the project's sliced-training reference body size",
    "dataset": "measured from the project's labelled objects",
    "derived": "computed from the settings above",
    "default": "the backend default",
}

# (tile_px or None, body_px, fraction or None) -> (label text, tooltip)
TileLabelFormatter = Callable[[int | None, float, float | None], tuple[str, str]]


def default_tile_label(
    tile_px: int | None, body_px: float, fraction: float | None
) -> tuple[str, str]:
    """Escalation-role resolved-tile text when the host supplies no formatter."""
    if tile_px:
        return (
            f"{tile_px} px ({body_px:.0f} px / {fraction:g})",
            f"Tile size {tile_px} px = {body_px:.0f} px body size / "
            f"{fraction:g} object scale.",
        )
    if fraction is None:
        return "full frame — tiling off.", ""
    return "full frame — no body size is known, so tiling is off.", ""


@dataclass(frozen=True)
class SliceWidgetCapabilities:
    """What a host lets the widget expose (decision 25).

    ``body_override``: the body size can be overridden (else display-only,
    except a 0 = unknown body, which always stays typeable -- I6).
    ``advanced_merge``: Advanced shows the merge policy/metric rows (display
    only; the merge threshold row is shown for ``infer_yolo`` regardless).
    ``tile_label_formatter``: the host's resolved-tile wording (escalation).
    ``fixed_overlap``: overlap is this constant, shown disabled (SAM2).
    ``full_frame_pass`` / ``execution_knobs``: TrackerKit-only Advanced rows
    (extra full-frame pass; tiles per call and memory budget).
    ``merge_threshold_row``: ``infer_yolo`` shows the merge threshold row
    (False where merge settings are profile-owned: TrackerKit).
    ``body_display_only``: the host owns the body elsewhere and only shows it
    (TrackerKit: the model's stamp/profile); never editable, not even an
    unknown 0, and the host's source badge is kept as given.
    """

    body_override: bool = True
    advanced_merge: bool = True
    tile_label_formatter: TileLabelFormatter | None = None
    fixed_overlap: float | None = None
    full_frame_pass: bool = False
    execution_knobs: bool = False
    body_display_only: bool = False
    merge_threshold_row: bool = True


def default_capabilities(role: str) -> SliceWidgetCapabilities:
    if role == "escalate_sam2":
        return SliceWidgetCapabilities(
            advanced_merge=False, fixed_overlap=SAM2_FIXED_OVERLAP_FALLBACK
        )
    if role == "escalate_sam3":
        return SliceWidgetCapabilities(advanced_merge=False)
    return SliceWidgetCapabilities()


def item_combo(items, tooltip: str) -> QComboBox:
    """A combo whose item text is the label, item data the enum (and tooltip)."""
    combo = QComboBox()
    for value, label, tip in items:
        combo.addItem(label, value)
        combo.setItemData(combo.count() - 1, tip, Qt.ItemDataRole.ToolTipRole)
    combo.setToolTip(tooltip)
    return combo


def double_spin(
    lo: float, hi: float, step: float, decimals: int, tooltip: str
) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setDecimals(decimals)
    spin.setRange(lo, hi)
    spin.setSingleStep(step)
    spin.setToolTip(tooltip)
    return spin


def int_spin(lo: int, hi: int, tooltip: str, special: str = "") -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(lo, hi)
    spin.setToolTip(tooltip)
    if special:
        spin.setSpecialValueText(special)
    return spin


def badge_label() -> QLabel:
    """A muted, small source badge (``dataset``, ``derived`` ...)."""
    label = QLabel()
    label.setObjectName("sliceSourceBadge")
    label.setStyleSheet("color: #8f969e; font-size: 11px; font-style: italic;")
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    return label


def muted_label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("color: #8f969e;")
    return label


def hbox(*widgets, stretch: bool = True) -> QWidget:
    """Pack widgets in one row without margins (a composite grid cell)."""
    holder = QWidget()
    layout = QHBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    for widget in widgets:
        layout.addWidget(widget)
    if stretch:
        layout.addStretch(1)
    return holder


def set_badge(label: QLabel, source: str) -> None:
    label.setText(source)
    label.setToolTip(f"Source: {SOURCE_DESCRIPTIONS.get(source, source)}.")


def widget_stylesheet(text_color: str | None, *, bare: bool) -> str:
    """Widget-scoped styling that reads correctly under any host theme.

    Constrained controls must LOOK disabled even where the host theme styles
    inputs but not their ``:disabled`` state (DetectKit's dark theme). Tool
    buttons take the host's label colour, which a theme sets through a
    stylesheet the buttons themselves do not match.
    """
    style = f"QToolButton {{ color: {text_color}; }} " if text_color else ""
    style += (
        "QAbstractSpinBox:disabled, QComboBox:disabled, QLineEdit:disabled,"
        " QCheckBox:disabled, QToolButton:disabled { color: #a0a5ab; }"
        " QToolButton#sliceOverlapRaise { border: 1px solid #8f969e;"
        " border-radius: 3px; padding: 1px 8px; background: transparent; }"
    )
    if bare:
        style += " QGroupBox { border: 0; margin-top: 0; padding: 0; }"
    return style


def tile_text(
    *,
    mode: str,
    fractions: list[float],
    body: float,
    imgsz: int,
    custom_wh: tuple[int, int],
    training: bool,
) -> str:
    """The derived tile-size label for the non-escalation roles."""
    if mode == "custom":
        w = custom_wh[0] or imgsz
        h = custom_wh[1] or imgsz
        return f"→ {w} × {h} px"
    if mode == "auto_model" or not fractions:
        return f"→ {imgsz} × {imgsz} px (model input)"
    if body <= 0:
        if training:
            return "→ body size ÷ scale, measured at build"
        return f"→ {imgsz} × {imgsz} px until a body size is known"
    sizes = sorted(
        {
            tile_size_for_mode(
                geometry_mode="auto_object",
                imgsz=imgsz,
                reference_body_px=body,
                object_tile_fraction=fraction,
                slice_width=0,
                slice_height=0,
            )[0]
            for fraction in fractions
        }
    )
    if len(fractions) == 1:
        return f"→ {sizes[0]} × {sizes[0]} px ({body:.0f} px ÷ {fractions[0]:g})"
    span = f"{sizes[0]}–{sizes[-1]}" if len(sizes) > 1 else f"{sizes[0]}"
    return f"→ {span} px over {len(fractions)} scales"


def whole_animal_minimum(
    fractions, *, decimals: int, ceiling: float
) -> tuple[float, bool] | None:
    """(max(scale) + margin, capped) rounded UP to ``decimals``.

    F7's whole-animal MINIMUM overlap -- never a recommendation. ``capped``
    when it exceeds OVERLAP_MAX and so is unreachable; None when nothing is
    tiled by scale. resolve_overlap's rule without its logging: this runs on
    every keystroke.
    """
    usable = [float(f) for f in fractions if f > 0]
    if not usable:
        return None
    raw = max(FRACTION_MIN, min(max(usable), FRACTION_MAX)) + OVERLAP_MARGIN
    capped = raw > OVERLAP_MAX + 1e-9
    scale = 10**decimals
    rounded = math.ceil(min(raw, OVERLAP_MAX) * scale - 1e-9) / scale
    return min(rounded, ceiling), capped
