"""Control construction and row tables for the shared SAHI settings widget.

Split out of ``slice_settings`` (size guideline); behaviour lives there.
Each function takes the widget ``w`` and builds/reads its attributes.
Shared layer: imports only Qt and ``utils`` (never an app layer).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QToolButton,
    QWidget,
)

from hydra_suite.utils.tiling_spec import FRACTION_MAX, FRACTION_MIN, OVERLAP_MAX

from .slice_settings_parts import (
    BODY_PX_MAX,
    ESCALATE_ROLES,
    FRAGMENT_ITEMS,
    GEOMETRY_ITEMS,
    MERGE_METRIC_ITEMS,
    MERGE_POLICY_ITEMS,
    SAM3_TRAIN_OVERLAP_MAX,
    SEAM_MARGIN_MAX_PX,
    SLICE_SIZE_MAX,
    SOURCE_DESCRIPTIONS,
    TRAIN_ROLES,
    ElidingLabel,
    badge_label,
    double_spin,
    hbox,
    int_spin,
    item_combo,
    muted_label,
    stacked,
)
from .tile_layout_preview import TileLayoutPreview

# Advanced rows, collapsed by default.
ADVANCED = (
    "merge_policy",
    "merge_metric",
    "merge",
    "seam",
    "min_area",
    "fragment",
    "negative",
    "keep_empty",
    "full_frame",
    "full_pass",
    "balance",
    "balance_power",
    "tile_batch",
    "memory",
)
FULL_WIDTH = ("enabled", "keep_empty", "full_frame", "full_pass", "balance")

ROLE_EXTRAS = {
    "train_yolo": {
        "negative_tile_fraction": "spin_slice_negative",
        "full_frame_mix": "chk_slice_full_frame_mix",
        "balance_multiscale_loss": "chk_slice_balance_loss",
        "balance_multiscale_loss_power": "spin_slice_balance_power",
        "min_area_ratio": "spin_slice_min_area",
    },
    "train_sam3": {
        "object_tile_fraction": "spin_slice_object_fraction",
        "keep_empty_tiles": "chk_slice_keep_empty",
        "full_frame_mix": "chk_slice_full_frame_mix",
        "min_area_ratio": "spin_slice_min_area",
        "tile_overlap": "spin_slice_overlap",
    },
    "infer_yolo": {
        "merge_threshold": "spin_slice_merge",
        "perform_standard_pred": "chk_slice_full_frame_pass",
        "tile_batch_size": "spin_slice_tile_batch",
        "memory_budget_mib": "spin_slice_memory_budget",
    },
    "escalate_sam3": {
        "seam_margin_px": "spin_slice_seam_margin",
        "merge_iou": "spin_slice_merge",
    },
    "escalate_sam2": {},
}


def build_controls(w) -> None:
    role = w._role
    # Bottom preview = a narrow host: derived notes go under their controls
    # and elide rather than widen the block.
    # Compact layout (S7): the notes fold into one summary line that elides.
    paired = w._caps.layout == "compact"
    compact = w._caps.preview_position == "bottom" or paired
    w.chk_slice_enabled = QCheckBox(
        "Enable sliced training + preview"
        if role == "train_yolo"
        else "Enable sliced inference"
    )
    w.chk_slice_enabled.setToolTip(
        "Generate sliced training examples and use the same tile geometry for "
        "DetectKit preview inference."
        if role == "train_yolo"
        else "Run the detector on overlapping tiles instead of one resized frame."
    )
    w.combo_slice_profile = QComboBox()
    w.combo_slice_profile.setToolTip(
        "Training geometry stamped on the model, a calibrated profile, or Custom."
    )
    # No status label here: the profile's status is host prose (TrackerKit
    # shows it as its own full-width row); a word-wrapped label inside a grid
    # cell would overlap its neighbours.
    w.combo_slice_geometry = item_combo(
        GEOMETRY_ITEMS,
        "How tile size is chosen: fit to the animal, the model input, or a "
        "custom size.",
    )

    w.txt_slice_scales = QLineEdit()
    w.txt_slice_scales.setPlaceholderText("e.g. 0.05, 0.1, 0.15")

    if role == "train_sam3":
        w.spin_slice_object_fraction = double_spin(
            0.0,
            1.0,
            0.001,
            4,
            "Single-scale object size as a fraction of the 1008px SAM3 input. "
            "Used when no scale set is listed above, or when the tile strategy "
            "is not 'Fit to animal size'.",
        )
    elif role in ESCALATE_ROLES:
        w.spin_slice_object_fraction = double_spin(
            0.0,
            FRACTION_MAX,
            0.01,
            3,
            "Tile size = body size / this object scale. 0 = full frame (no "
            "tiling). The default is a starting guess, not a tuned value — "
            "calibrate against your own labelled frames to fit it.",
        )
        w.spin_slice_object_fraction.setSpecialValueText("full frame (no tiling)")
    else:
        w.spin_slice_object_fraction = double_spin(
            FRACTION_MIN,
            FRACTION_MAX,
            0.005,
            3,
            "Object size as a fraction of the tile (the model input). Larger "
            "fractions use smaller tiles and can make high-resolution "
            "inference much slower. Used with 'Fit to animal size'.",
        )
    w.lbl_slice_scale_px = muted_label(eliding=compact)

    w.spin_slice_body = double_spin(
        0.0,
        BODY_PX_MAX,
        5.0,
        1,
        "The typical longest side of one animal, in source-image pixels. "
        "Tile size = this / object scale.",
    )
    w.spin_slice_body.setSpecialValueText(
        "unknown (tiling off)"
        if role in ESCALATE_ROLES
        else "unknown (model input tiles)"
    )
    w.chk_slice_body_override = QCheckBox("Override")
    w.chk_slice_body_override.setToolTip(
        "Edit a body size that was measured or stamped. Unchecking restores "
        "the derived value."
    )
    w.lbl_slice_body_badge = badge_label()
    w.auto_reference_note = QLabel()
    w.auto_reference_note.setWordWrap(True)
    w.auto_reference_note.setStyleSheet("color: #8f969e;")

    tile_tip = (
        "Custom tile {} in source-image pixels. Zero uses the model input "
        "size. Editable only with 'Custom tile size'."
    )
    w.spin_slice_tile_w = int_spin(
        0, SLICE_SIZE_MAX, tile_tip.format("width"), "model input"
    )
    w.spin_slice_tile_h = int_spin(
        0, SLICE_SIZE_MAX, tile_tip.format("height"), "model input"
    )
    w.lbl_slice_tile_size = ElidingLabel() if compact else QLabel()
    # Never word-wrapped: a wrapped label's height-for-width is not part of
    # the grid's minimum, so it would overlap its neighbours. Hosts keep the
    # visible text short (explicit newlines are fine) and put prose in the
    # tooltip.
    w.lbl_slice_tile_size.setWordWrap(False)
    if role in ESCALATE_ROLES:
        w.lbl_slice_tile_size.setMinimumWidth(180)
    w.lbl_slice_tile_badge = badge_label()

    w.spin_slice_overlap = double_spin(
        0.0,
        SAM3_TRAIN_OVERLAP_MAX if role == "train_sam3" else OVERLAP_MAX,
        0.01 if role == "train_sam3" else 0.05,
        3 if role == "train_sam3" else 2,
        "Fraction shared by neighbouring tiles. More overlap protects objects "
        "at tile edges but creates more inference work.",
    )
    w.lbl_slice_overlap_minimum = muted_label(eliding=compact)
    w.btn_slice_overlap_raise = QToolButton()
    w.btn_slice_overlap_raise.setText("Raise")
    w.btn_slice_overlap_raise.setObjectName("sliceOverlapRaise")
    w.btn_slice_overlap_raise.setToolTip(
        "Raise the overlap to the whole-animal minimum (largest object scale + "
        "margin), so every animal lies whole inside at least one tile."
    )

    # Compact layout: tile px, object-scale px and the overlap note on one
    # muted line under the grid (full text in the tooltip).
    w.lbl_slice_summary = muted_label(eliding=True)
    w.lbl_slice_summary.setObjectName("sliceSummary")

    w.btn_slice_advanced = QToolButton()
    w.btn_slice_advanced.setText("Advanced")
    w.btn_slice_advanced.setCheckable(True)
    w.btn_slice_advanced.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    w.btn_slice_advanced.setArrowType(Qt.ArrowType.RightArrow)
    w.btn_slice_advanced.setAutoRaise(True)
    w.btn_slice_advanced.setToolTip("Show or hide the advanced SAHI settings.")

    w.combo_slice_merge_policy = item_combo(
        MERGE_POLICY_ITEMS,
        "How duplicate detections from neighbouring tiles are merged. Set by "
        "the model's SAHI profile (default greedy NMM); not stored here.",
    )
    w.combo_slice_merge_metric = item_combo(
        MERGE_METRIC_ITEMS,
        "Overlap measure used to find duplicates. Set by the model's SAHI "
        "profile (default IoS); not stored here.",
    )
    if role == "escalate_sam3":
        w.spin_slice_merge = double_spin(
            0.05,
            0.95,
            0.05,
            2,
            "Polygon IoU above which two masks from neighbouring tiles are "
            "the same animal.",
        )
    else:
        w.spin_slice_merge = double_spin(
            0.0,
            1.0,
            0.05,
            2,
            "Overlap threshold used to merge duplicate predictions from "
            "neighbouring tiles during preview inference.",
        )
    w.spin_slice_seam_margin = int_spin(
        0,
        SEAM_MARGIN_MAX_PX,
        "Masks touching a tile edge within this many pixels are treated as "
        "cut by the seam when merging.",
    )
    w.spin_slice_min_area = double_spin(
        0.0,
        1.0,
        0.05,
        2,
        (
            (
                "Minimum fraction of a labelled object's original area that must "
                "lie in a tile for that label to stay a full training target. "
                "Below this floor SAM3 keeps the fragment but marks it "
                "'is_crowd', which downgrades the tile's exhaustiveness rather "
                "than dropping the label."
            )
            if role == "train_sam3"
            else (
                "Minimum fraction of a labelled object's original area that must "
                "lie in a tile to keep that label. For example, 0.10 keeps labels "
                "with at least 10%."
            )
        ),
    )
    w.combo_slice_fragment_policy = item_combo(
        FRAGMENT_ITEMS,
        "What happens to a fragment below the minimum area. Fixed by the "
        "training backend.",
    )
    w.spin_slice_negative = double_spin(
        0.0,
        1.0,
        0.05,
        2,
        "Sampling probability for background-only tiles. For example, 0.15 "
        "keeps 15% of empty tiles.",
    )
    w.chk_slice_keep_empty = QCheckBox("Keep empty tiles")
    w.chk_slice_keep_empty.setToolTip(
        "Keep tiles that contain no labelled object. SAM3 uses them as "
        "negative evidence rather than sampling a fraction of them."
    )
    w.chk_slice_full_frame_mix = QCheckBox("Mix full frames")
    w.chk_slice_full_frame_mix.setToolTip(
        "Include unsliced full-frame examples alongside tiles so the model "
        "retains global context."
    )
    w.chk_slice_full_frame_pass = QCheckBox("Extra full-frame pass")
    w.chk_slice_full_frame_pass.setToolTip(
        "Also run the detector once on the whole frame and merge it with the "
        "tiles (catches animals larger than a tile)."
    )
    w.chk_slice_balance_loss = QCheckBox("Balance multi-scale training loss")
    w.chk_slice_balance_loss.setToolTip(
        "Keep every generated tile once per epoch, but normalize the detector "
        "loss so a scale that creates more tiles cannot dominate training."
    )
    w.spin_slice_balance_power = double_spin(
        0.0,
        1.0,
        0.1,
        2,
        "0.5 uses square-root inverse-frequency balancing; 1.0 gives exact "
        "per-scale balance. Full-frame examples keep their normal weight.",
    )
    w.spin_slice_tile_batch = int_spin(
        0, 4096, "Tiles sent to the detector per call. 0 = automatic.", "auto"
    )
    w.spin_slice_memory_budget = int_spin(
        0,
        1 << 20,
        "GPU memory budget for one tile batch, in MiB. 0 = automatic.",
        "auto",
    )
    w.spin_slice_memory_budget.setSuffix(" MiB")
    for control in (
        w.combo_slice_profile,
        w.combo_slice_geometry,
        w.txt_slice_scales,
        w.spin_slice_object_fraction,
        w.spin_slice_body,
        w.spin_slice_overlap,
        w.combo_slice_merge_policy,
        w.combo_slice_merge_metric,
        w.spin_slice_merge,
        w.spin_slice_seam_margin,
        w.spin_slice_min_area,
        w.combo_slice_fragment_policy,
        w.spin_slice_negative,
        w.spin_slice_balance_power,
        w.spin_slice_tile_batch,
        w.spin_slice_memory_budget,
    ):
        # Compact: two fields share a row, so each may give way further.
        control.setMinimumWidth(72 if paired else 120)
        control.setMaximumWidth(220)
    for spin in (w.spin_slice_tile_w, w.spin_slice_tile_h):
        spin.setMinimumWidth(72 if paired else 112)
        spin.setMaximumWidth(140)
        if paired:
            # Half-width column: the short form of "model input" (tooltip
            # says what 0 means).
            spin.setSpecialValueText("input")
    w.preview = TileLayoutPreview()
    w.preview.set_bottom_layout(w._caps.preview_position == "bottom", compact=paired)
    if role == "infer_yolo":
        w.preview.set_body_notes(
            "uses the body size", "illustrative until a body size is known"
        )
    w._refresh_scales_tooltip()


def row_specs(w) -> list[tuple[str, str | None, QWidget, QLabel | None]]:
    """(key, label, control, badge) in display order."""
    role = w._role
    times = muted_label("×")
    w._tile_spins = hbox(w.spin_slice_tile_w, times, w.spin_slice_tile_h, stretch=False)
    paired = w._caps.layout == "compact"
    if paired:
        # S7: badges ride inside their field's cell; the derived notes go to
        # the summary line the widget builds under the grid.
        # Fields fill their half-width column (up to their maximum), so the
        # right edges of each column line up.
        w._stacked_rows = set()
        # The tile badge leads the summary line instead (no room here).
        tile_cell = hbox(w._tile_spins, stretch=False)
        object_cell = hbox(w.spin_slice_object_fraction, stretch=False)
        overlap_cell = hbox(w.spin_slice_overlap, stretch=False)
    elif w._caps.preview_position == "bottom":
        # Each derived note on its own line under its control (S6).
        w._stacked_rows = {"object_fraction", "tile", "overlap"}
        tile_cell = stacked(hbox(w._tile_spins), w.lbl_slice_tile_size)
        object_cell = stacked(hbox(w.spin_slice_object_fraction), w.lbl_slice_scale_px)
        overlap_cell = stacked(
            hbox(w.spin_slice_overlap, w.btn_slice_overlap_raise),
            w.lbl_slice_overlap_minimum,
        )
    else:
        w._stacked_rows = set()
        tile_cell = hbox(w._tile_spins, w.lbl_slice_tile_size)
        object_cell = hbox(w.spin_slice_object_fraction, w.lbl_slice_scale_px)
        overlap_cell = hbox(
            w.spin_slice_overlap,
            w.lbl_slice_overlap_minimum,
            w.btn_slice_overlap_raise,
        )
    body_badge, tile_badge = w.lbl_slice_body_badge, w.lbl_slice_tile_badge
    if role in TRAIN_ROLES:
        body_parts = [w.auto_reference_note]
    else:
        body_parts = [w.spin_slice_body, w.chk_slice_body_override]
    if paired:
        body_parts.append(body_badge)
        body_badge = tile_badge = None
    body_cell = hbox(*body_parts, stretch=role not in TRAIN_ROLES and not paired)
    object_label = (
        "Single-scale object fraction" if role == "train_sam3" else "Object scale"
    )
    return [
        ("enabled", None, w.chk_slice_enabled, None),
        (
            "profile",
            "Profile",
            hbox(w.combo_slice_profile, stretch=not paired),
            None,
        ),
        ("mode", "Tile strategy", w.combo_slice_geometry, None),
        ("targets", "Object scales", w.txt_slice_scales, None),
        ("object_fraction", object_label, object_cell, None),
        ("body", "Body size", body_cell, body_badge),
        (
            "tile",
            "Resolved tile" if role in ESCALATE_ROLES else "Tile size",
            tile_cell,
            tile_badge,
        ),
        ("overlap", "Tile overlap", overlap_cell, None),
        ("advanced", None, w.btn_slice_advanced, None),
        ("merge_policy", "Merge policy", w.combo_slice_merge_policy, None),
        ("merge_metric", "Merge metric", w.combo_slice_merge_metric, None),
        (
            "merge",
            "Merge IoU" if role == "escalate_sam3" else "Merge threshold",
            w.spin_slice_merge,
            None,
        ),
        ("seam", "Seam margin (px)", w.spin_slice_seam_margin, None),
        (
            "min_area",
            "Minimum retained object area",
            w.spin_slice_min_area,
            None,
        ),
        ("fragment", "Below the floor", w.combo_slice_fragment_policy, None),
        (
            "negative",
            "Empty-tile sampling fraction",
            w.spin_slice_negative,
            None,
        ),
        ("keep_empty", None, w.chk_slice_keep_empty, None),
        ("full_frame", None, w.chk_slice_full_frame_mix, None),
        ("full_pass", None, w.chk_slice_full_frame_pass, None),
        ("balance", None, w.chk_slice_balance_loss, None),
        ("balance_power", "Balance strength", w.spin_slice_balance_power, None),
        ("tile_batch", "Tiles per call", w.spin_slice_tile_batch, None),
        ("memory", "Tile memory budget", w.spin_slice_memory_budget, None),
    ]


def role_keys(w) -> set[str]:
    role, caps = w._role, w._caps
    keys = {"body", "tile", "overlap"}
    if role in ("infer_yolo", "train_yolo"):
        keys.add("enabled")
    if role == "infer_yolo":
        keys |= {"profile", "object_fraction"}
        if caps.merge_threshold_row:
            keys.add("merge")
        if caps.advanced_merge:
            keys |= {"merge_policy", "merge_metric"}
        if caps.full_frame_pass:
            keys.add("full_pass")
        if caps.execution_knobs:
            keys |= {"tile_batch", "memory"}
    if role in TRAIN_ROLES:
        keys |= {"mode", "targets", "min_area", "fragment", "full_frame"}
    if role == "infer_yolo":
        keys.add("mode")
    if role == "train_yolo":
        keys |= {"merge", "negative", "balance", "balance_power"}
    if role == "train_sam3":
        keys |= {"object_fraction", "keep_empty"}
    if role in ESCALATE_ROLES:
        keys.add("object_fraction")
    if role == "escalate_sam3":
        keys |= {"merge", "seam"}
    if keys & set(ADVANCED):
        keys.add("advanced")
    return keys


def refresh_overlap_minimum(w) -> None:
    label, button = w.lbl_slice_overlap_minimum, w.btn_slice_overlap_raise
    result = w._whole_animal_minimum()
    tiling_on = w.chk_slice_enabled.isChecked() if "enabled" in w._role_rows else True
    if result is None:
        label.setText("")
        button.setVisible(False)
        return
    minimum, capped = result
    decimals = w.spin_slice_overlap.decimals()
    shown = f"{minimum:.{decimals}f}" + (", capped" if capped else "")
    below = w.spin_slice_overlap.value() < minimum - 0.5 * 10**-decimals
    if below and not tiling_on:
        # Nothing is tiled: no warning, no button for an inert setting.
        label.setText("")
        button.setVisible(False)
        return
    source = w._sources.get("overlap", "user")
    if below and source not in ("user", "override", "default"):
        # A measured/deliberate overlap (profile, stamp, saved session) is
        # never nudged (decisions 22/22a): info, no Raise.
        origin = w._source_notes.get("overlap") or SOURCE_DESCRIPTIONS.get(
            source, source
        )
        label.setText(f"below whole-animal minimum ({shown}) — set by {origin}")
        label.setStyleSheet("color: #8f969e;")
        label.setToolTip(
            "This overlap came from a calibration or the model's stamp and is "
            "kept as measured. An animal at a tile seam may be cut in every "
            "tile; edit the overlap to take it over."
        )
        button.setVisible(False)
        return
    if below:
        label.setText(f"Below whole-animal minimum ({shown})")
        label.setStyleSheet("color: #e0943a;")
        label.setToolTip(
            "With less overlap than an animal's share of a tile (the largest "
            "object scale, or body size / tile size), an animal at a tile "
            "seam can be cut in every tile. Your overlap is kept until you "
            "raise it."
        )
        button.setText(f"Raise to {minimum:.{decimals}f}")
    else:
        label.setText(f"≥ whole-animal minimum ({shown})")
        label.setStyleSheet("color: #8f969e;")
        label.setToolTip(
            "Every animal fits whole inside at least one tile at this overlap."
        )
    button.setVisible(below)
