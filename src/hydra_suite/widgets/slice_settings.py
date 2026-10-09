"""One shared SAHI settings widget for every host (spec §5, S4 Task 15).

The widget's state is the S1 contract (:class:`TilingSpec`) plus a
role-specific ``extras`` dict. Ownership model (plan decision 23): the widget
writes its own controls ONLY inside :meth:`SliceSettingsWidget.set_spec`
(signals blocked, then derived labels refreshed). Derived values (tile size
outside Custom, the suggested overlap, the resolved tile) are shown in
LABELS, never written into a host-persisted spin. A user edit emits
``field_changed(field)``.

Shared layer: imports only Qt and ``utils`` (never an app layer).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QToolButton,
    QWidget,
)

from hydra_suite.utils.slice_geometry import tile_size_for_mode
from hydra_suite.utils.tiling_spec import (
    BACKEND_DEFAULTS,
    DEFAULT_YOLO_IMGSZ,
    FRACTION_MAX,
    FRACTION_MIN,
    OVERLAP_MAX,
    TilingSpec,
    resolve_overlap,
)

from .slice_settings_parts import (
    BODY_PX_MAX,
    ESCALATE_ROLES,
    FRAGMENT_ITEMS,
    GEOMETRY_ITEMS,
    MERGE_METRIC_ITEMS,
    MERGE_POLICY_ITEMS,
    ROLE_BACKEND,
    ROLES,
    SAM3_TRAIN_OVERLAP_MAX,
    SEAM_MARGIN_MAX_PX,
    SLICE_SIZE_MAX,
    SOURCE_DESCRIPTIONS,
    TRAIN_ROLES,
    SliceWidgetCapabilities,
    badge_label,
    default_capabilities,
    default_tile_label,
    double_spin,
    hbox,
    int_spin,
    item_combo,
    muted_label,
    set_badge,
)
from .tile_layout_preview import TileLayoutPreview

__all__ = ["ROLES", "SliceSettingsWidget", "SliceWidgetCapabilities"]

# Advanced rows, collapsed by default.
_ADVANCED = (
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
_FULL_WIDTH = ("enabled", "keep_empty", "full_frame", "full_pass", "balance")


class SliceSettingsWidget(QGroupBox):
    """SAHI settings with TrackerKit's vocabulary for one host ``role``."""

    field_changed = Signal(str)

    def __init__(
        self,
        parent=None,
        *,
        role: str,
        title: str | None = None,
        capabilities: SliceWidgetCapabilities | None = None,
    ) -> None:
        if role not in ROLES:
            raise ValueError(f"unknown SAHI widget role: {role!r}")
        super().__init__(title or "", parent)
        self._role = role
        self._caps = capabilities or default_capabilities(role)
        self._base = TilingSpec.defaults(ROLE_BACKEND[role])
        self._passthrough: dict[str, Any] = {}
        self._model_input_size = DEFAULT_YOLO_IMGSZ
        self._body_derived = (0.0, "default")
        self._body_source = "default"
        self._sources: dict[str, str] = {}
        self._advanced_expanded = False
        self._profile_row_shown = False
        self._loading = False
        if not title:
            self.setFlat(True)
            self.setStyleSheet("QGroupBox { border: 0; margin-top: 0; padding: 0; }")

        self._build_controls()
        self._build_layout(bare=not title)
        self._wire()
        self._apply_role_defaults()
        self._refresh()

    # ------------------------------------------------------------------ build

    def _build_controls(self) -> None:
        role = self._role
        self.chk_slice_enabled = QCheckBox(
            "Enable sliced training + preview"
            if role == "train_yolo"
            else "Sliced inference (SAHI)"
        )
        self.chk_slice_enabled.setToolTip(
            "Generate sliced training examples and use the same tile geometry for "
            "DetectKit preview inference."
            if role == "train_yolo"
            else "Run the detector on overlapping tiles instead of one resized frame."
        )
        self.combo_slice_profile = QComboBox()
        self.combo_slice_profile.setToolTip(
            "Training geometry stamped on the model, a calibrated profile, or Custom."
        )
        self.lbl_slice_profile_status = muted_label()
        self.lbl_slice_profile_status.setWordWrap(True)
        self.combo_slice_geometry = item_combo(
            GEOMETRY_ITEMS,
            "How tile size is chosen: fit to the animal, the model input, or a "
            "custom size.",
        )

        self.txt_slice_scales = QLineEdit()
        self.txt_slice_scales.setPlaceholderText("e.g. 0.05, 0.1, 0.15")

        if role == "train_sam3":
            self.spin_slice_object_fraction = double_spin(
                0.0,
                1.0,
                0.001,
                4,
                "Single-scale object size as a fraction of the 1008px SAM3 input. "
                "Used when no scale set is listed above, or when the tile strategy "
                "is not 'Fit to animal size'.",
            )
        elif role in ESCALATE_ROLES:
            self.spin_slice_object_fraction = double_spin(
                0.0,
                FRACTION_MAX,
                0.01,
                3,
                "Tile size = body size / this object scale. 0 = full frame (no "
                "tiling). The default is a starting guess, not a tuned value — "
                "calibrate against your own labelled frames to fit it.",
            )
            self.spin_slice_object_fraction.setSpecialValueText(
                "full frame (no tiling)"
            )
        else:
            self.spin_slice_object_fraction = double_spin(
                FRACTION_MIN,
                FRACTION_MAX,
                0.005,
                3,
                "Object size as a fraction of the tile (the model input). Larger "
                "fractions use smaller tiles and can make high-resolution "
                "inference much slower. Used with 'Fit to animal size'.",
            )
        self.lbl_slice_scale_px = muted_label()

        self.spin_slice_body = double_spin(
            0.0,
            BODY_PX_MAX,
            5.0,
            1,
            "The typical longest side of one animal, in source-image pixels. "
            "Tile size = this / object scale.",
        )
        self.spin_slice_body.setSpecialValueText(
            "unknown (tiling off)"
            if role in ESCALATE_ROLES
            else "unknown (model input tiles)"
        )
        self.chk_slice_body_override = QCheckBox("Override")
        self.chk_slice_body_override.setToolTip(
            "Edit a body size that was measured or stamped. Unchecking restores "
            "the derived value."
        )
        self.lbl_slice_body_badge = badge_label()
        self.auto_reference_note = QLabel()
        self.auto_reference_note.setWordWrap(True)
        self.auto_reference_note.setStyleSheet("color: #8f969e;")

        tile_tip = (
            "Custom tile {} in source-image pixels. Zero uses the model input "
            "size. Editable only with 'Custom tile size'."
        )
        self.spin_slice_tile_w = int_spin(
            0, SLICE_SIZE_MAX, tile_tip.format("width"), "model input"
        )
        self.spin_slice_tile_h = int_spin(
            0, SLICE_SIZE_MAX, tile_tip.format("height"), "model input"
        )
        self.lbl_slice_tile_size = QLabel()
        self.lbl_slice_tile_size.setWordWrap(role in ESCALATE_ROLES)
        if role in ESCALATE_ROLES:
            self.lbl_slice_tile_size.setMinimumWidth(180)
        self.lbl_slice_tile_badge = badge_label()

        self.spin_slice_overlap = double_spin(
            0.0,
            SAM3_TRAIN_OVERLAP_MAX if role == "train_sam3" else OVERLAP_MAX,
            0.01 if role == "train_sam3" else 0.05,
            3 if role == "train_sam3" else 2,
            "Fraction shared by neighbouring tiles. More overlap protects objects "
            "at tile edges but creates more inference work.",
        )
        self.lbl_slice_overlap_suggested = muted_label()
        self.btn_slice_overlap_use_suggested = QPushButton("Use suggested")
        self.btn_slice_overlap_use_suggested.setToolTip(
            "Set the overlap to the largest object scale + margin, which keeps "
            "every animal whole inside at least one tile."
        )

        self.btn_slice_advanced = QToolButton()
        self.btn_slice_advanced.setText("Advanced")
        self.btn_slice_advanced.setCheckable(True)
        self.btn_slice_advanced.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.btn_slice_advanced.setArrowType(Qt.ArrowType.RightArrow)
        self.btn_slice_advanced.setAutoRaise(True)
        self.btn_slice_advanced.setToolTip("Show or hide the advanced SAHI settings.")

        self.combo_slice_merge_policy = item_combo(
            MERGE_POLICY_ITEMS,
            "How duplicate detections from neighbouring tiles are merged. Set by "
            "the model's SAHI profile (default greedy NMM); not stored here.",
        )
        self.combo_slice_merge_metric = item_combo(
            MERGE_METRIC_ITEMS,
            "Overlap measure used to find duplicates. Set by the model's SAHI "
            "profile (default IoS); not stored here.",
        )
        if role == "escalate_sam3":
            self.spin_slice_merge = double_spin(
                0.05,
                0.95,
                0.05,
                2,
                "Polygon IoU above which two masks from neighbouring tiles are "
                "the same animal.",
            )
        else:
            self.spin_slice_merge = double_spin(
                0.0,
                1.0,
                0.05,
                2,
                "Overlap threshold used to merge duplicate predictions from "
                "neighbouring tiles during preview inference.",
            )
        self.spin_slice_seam_margin = int_spin(
            0,
            SEAM_MARGIN_MAX_PX,
            "Masks touching a tile edge within this many pixels are treated as "
            "cut by the seam when merging.",
        )
        self.spin_slice_min_area = double_spin(
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
        self.combo_slice_fragment_policy = item_combo(
            FRAGMENT_ITEMS,
            "What happens to a fragment below the minimum area. Fixed by the "
            "training backend.",
        )
        self.spin_slice_negative = double_spin(
            0.0,
            1.0,
            0.05,
            2,
            "Sampling probability for background-only tiles. For example, 0.15 "
            "keeps 15% of empty tiles.",
        )
        self.chk_slice_keep_empty = QCheckBox("Keep empty tiles")
        self.chk_slice_keep_empty.setToolTip(
            "Keep tiles that contain no labelled object. SAM3 uses them as "
            "negative evidence rather than sampling a fraction of them."
        )
        self.chk_slice_full_frame_mix = QCheckBox("Mix full frames")
        self.chk_slice_full_frame_mix.setToolTip(
            "Include unsliced full-frame examples alongside tiles so the model "
            "retains global context."
        )
        self.chk_slice_full_frame_pass = QCheckBox("Extra full-frame pass")
        self.chk_slice_full_frame_pass.setToolTip(
            "Also run the detector once on the whole frame and merge it with the "
            "tiles (catches animals larger than a tile)."
        )
        self.chk_slice_balance_loss = QCheckBox("Balance multi-scale training loss")
        self.chk_slice_balance_loss.setToolTip(
            "Keep every generated tile once per epoch, but normalize the detector "
            "loss so a scale that creates more tiles cannot dominate training."
        )
        self.spin_slice_balance_power = double_spin(
            0.0,
            1.0,
            0.1,
            2,
            "0.5 uses square-root inverse-frequency balancing; 1.0 gives exact "
            "per-scale balance. Full-frame examples keep their normal weight.",
        )
        self.spin_slice_tile_batch = int_spin(
            0, 4096, "Tiles sent to the detector per call. 0 = automatic.", "auto"
        )
        self.spin_slice_memory_budget = int_spin(
            0,
            1 << 20,
            "GPU memory budget for one tile batch, in MiB. 0 = automatic.",
            "auto",
        )
        self.spin_slice_memory_budget.setSuffix(" MiB")
        for control in (
            self.combo_slice_profile,
            self.combo_slice_geometry,
            self.txt_slice_scales,
            self.spin_slice_object_fraction,
            self.spin_slice_body,
            self.spin_slice_overlap,
            self.combo_slice_merge_policy,
            self.combo_slice_merge_metric,
            self.spin_slice_merge,
            self.spin_slice_seam_margin,
            self.spin_slice_min_area,
            self.combo_slice_fragment_policy,
            self.spin_slice_negative,
            self.spin_slice_balance_power,
            self.spin_slice_tile_batch,
            self.spin_slice_memory_budget,
        ):
            control.setMinimumWidth(120)
            control.setMaximumWidth(220)
        for spin in (self.spin_slice_tile_w, self.spin_slice_tile_h):
            spin.setMinimumWidth(112)
            spin.setMaximumWidth(140)
        self.preview = TileLayoutPreview()
        if role == "infer_yolo":
            self.preview.set_body_notes(
                "uses the body size", "illustrative until a body size is known"
            )
        self._refresh_scales_tooltip()

    def _row_specs(self) -> list[tuple[str, str | None, QWidget, QLabel | None]]:
        """(key, label, control, badge) in display order."""
        role = self._role
        times = muted_label("×")
        self._tile_spins = hbox(
            self.spin_slice_tile_w, times, self.spin_slice_tile_h, stretch=False
        )
        tile_cell = hbox(self._tile_spins, self.lbl_slice_tile_size)
        if role in TRAIN_ROLES:
            body_cell = hbox(self.auto_reference_note, stretch=False)
        else:
            body_cell = hbox(self.spin_slice_body, self.chk_slice_body_override)
        object_label = (
            "Single-scale object fraction" if role == "train_sam3" else "Object scale"
        )
        return [
            ("enabled", None, self.chk_slice_enabled, None),
            (
                "profile",
                "Profile",
                hbox(self.combo_slice_profile, self.lbl_slice_profile_status),
                None,
            ),
            ("mode", "Tile strategy", self.combo_slice_geometry, None),
            ("targets", "Object scales", self.txt_slice_scales, None),
            (
                "object_fraction",
                object_label,
                hbox(self.spin_slice_object_fraction, self.lbl_slice_scale_px),
                None,
            ),
            ("body", "Body size", body_cell, self.lbl_slice_body_badge),
            (
                "tile",
                "Resolved tile" if role in ESCALATE_ROLES else "Tile size",
                tile_cell,
                self.lbl_slice_tile_badge,
            ),
            (
                "overlap",
                "Tile overlap",
                hbox(
                    self.spin_slice_overlap,
                    self.lbl_slice_overlap_suggested,
                    self.btn_slice_overlap_use_suggested,
                ),
                None,
            ),
            ("advanced", None, self.btn_slice_advanced, None),
            ("merge_policy", "Merge policy", self.combo_slice_merge_policy, None),
            ("merge_metric", "Merge metric", self.combo_slice_merge_metric, None),
            (
                "merge",
                "Merge IoU" if role == "escalate_sam3" else "Merge threshold",
                self.spin_slice_merge,
                None,
            ),
            ("seam", "Seam margin (px)", self.spin_slice_seam_margin, None),
            (
                "min_area",
                "Minimum retained object area",
                self.spin_slice_min_area,
                None,
            ),
            ("fragment", "Below the floor", self.combo_slice_fragment_policy, None),
            (
                "negative",
                "Empty-tile sampling fraction",
                self.spin_slice_negative,
                None,
            ),
            ("keep_empty", None, self.chk_slice_keep_empty, None),
            ("full_frame", None, self.chk_slice_full_frame_mix, None),
            ("full_pass", None, self.chk_slice_full_frame_pass, None),
            ("balance", None, self.chk_slice_balance_loss, None),
            ("balance_power", "Balance strength", self.spin_slice_balance_power, None),
            ("tile_batch", "Tiles per call", self.spin_slice_tile_batch, None),
            ("memory", "Tile memory budget", self.spin_slice_memory_budget, None),
        ]

    def _role_keys(self) -> set[str]:
        role, caps = self._role, self._caps
        keys = {"body", "tile", "overlap"}
        if role in ("infer_yolo", "train_yolo"):
            keys.add("enabled")
        if role == "infer_yolo":
            keys |= {"profile", "object_fraction"}
            if caps.advanced_merge:
                keys |= {"merge_policy", "merge_metric", "merge"}
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
        if keys & set(_ADVANCED):
            keys.add("advanced")
        return keys

    def _build_layout(self, *, bare: bool) -> None:
        outer = QHBoxLayout(self)
        if bare:
            outer.setContentsMargins(0, 0, 0, 0)
        else:
            outer.setContentsMargins(14, 16, 14, 12)
        outer.setSpacing(18)
        controls = QWidget()
        grid = QGridLayout(controls)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(7)
        grid.setColumnStretch(1, 1)
        self._grid = grid
        self._role_rows = self._role_keys()
        # Legacy name kept for DetectKit tests: key -> (label, control).
        self._rows: dict[str, tuple[QLabel, QWidget]] = {}
        self._row_widgets: dict[str, list[QWidget]] = {}
        for row, (key, text, control, badge) in enumerate(self._row_specs()):
            widgets: list[QWidget] = [control]
            if text is None or key in _FULL_WIDTH or key == "advanced":
                grid.addWidget(control, row, 0, 1, 3)
            else:
                label = QLabel(text)
                label.setToolTip(self._label_tooltip(control))
                grid.addWidget(label, row, 0)
                grid.addWidget(control, row, 1)
                self._rows[key] = (label, control)
                widgets.append(label)
            if badge is not None:
                grid.addWidget(badge, row, 2)
                widgets.append(badge)
            self._row_widgets[key] = widgets
        grid.setRowStretch(len(self._row_widgets), 1)
        outer.addWidget(controls, 0)
        if self._role in ESCALATE_ROLES:
            self.preview.hide()
        else:
            outer.addWidget(self.preview, 1)

    @staticmethod
    def _label_tooltip(control: QWidget) -> str:
        if control.toolTip():
            return control.toolTip()
        tips = [child.toolTip() for child in control.findChildren(QWidget)]
        return next((tip for tip in tips if tip), "")

    def _wire(self) -> None:
        edits = (
            (self.chk_slice_enabled.toggled, "enabled"),
            (self.combo_slice_geometry.currentIndexChanged, "geometry_mode"),
            (self.txt_slice_scales.textChanged, "object_tile_fractions"),
            (
                self.spin_slice_object_fraction.valueChanged,
                (
                    "object_tile_fraction"
                    if self._role == "train_sam3"
                    else "object_tile_fractions"
                ),
            ),
            (self.spin_slice_body.valueChanged, "reference_body_px"),
            (self.chk_slice_body_override.toggled, "body_override"),
            (self.spin_slice_tile_w.valueChanged, "slice_width"),
            (self.spin_slice_tile_h.valueChanged, "slice_height"),
            (self.spin_slice_overlap.valueChanged, "overlap"),
            (self.combo_slice_merge_policy.currentIndexChanged, "merge_policy"),
            (self.combo_slice_merge_metric.currentIndexChanged, "merge_metric"),
            (
                self.spin_slice_merge.valueChanged,
                "merge_iou" if self._role == "escalate_sam3" else "merge_threshold",
            ),
            (self.spin_slice_seam_margin.valueChanged, "seam_margin_px"),
            (self.spin_slice_min_area.valueChanged, "min_area_ratio"),
            (self.combo_slice_fragment_policy.currentIndexChanged, "fragment_policy"),
            (self.spin_slice_negative.valueChanged, "negative_tile_fraction"),
            (self.chk_slice_keep_empty.toggled, "keep_empty_tiles"),
            (self.chk_slice_full_frame_mix.toggled, "full_frame_mix"),
            (self.chk_slice_full_frame_pass.toggled, "perform_standard_pred"),
            (self.chk_slice_balance_loss.toggled, "balance_multiscale_loss"),
            (
                self.spin_slice_balance_power.valueChanged,
                "balance_multiscale_loss_power",
            ),
            (self.spin_slice_tile_batch.valueChanged, "tile_batch_size"),
            (self.spin_slice_memory_budget.valueChanged, "memory_budget_mib"),
        )
        for signal, field in edits:
            signal.connect(lambda *_a, field=field: self._on_user_edit(field))
        self.btn_slice_overlap_use_suggested.clicked.connect(self._use_suggested)
        self.btn_slice_advanced.toggled.connect(self.set_advanced_expanded)

    def _apply_role_defaults(self) -> None:
        caps = self._caps
        if caps.fixed_overlap is not None:
            self.spin_slice_overlap.setValue(float(caps.fixed_overlap))
            self.spin_slice_overlap.setToolTip(
                "Fixed: owner tiles always overlap by this fraction."
            )
        self.chk_slice_body_override.setVisible(
            caps.body_override and self._role not in TRAIN_ROLES
        )
        self.btn_slice_overlap_use_suggested.setVisible(caps.fixed_overlap is None)
        if caps.fixed_overlap is not None:
            self.lbl_slice_overlap_suggested.setText("fixed")
            self.lbl_slice_overlap_suggested.setToolTip(
                "This backend's tiles always overlap by this constant."
            )
        self._tile_spins.setVisible(self._role not in ESCALATE_ROLES)
        self.lbl_slice_scale_px.setVisible(self._role == "infer_yolo")
        # Merge policy/metric are display-only here: no S4a host persists them
        # (DetectKit reads them from the model's profile).
        self.combo_slice_merge_policy.setEnabled(False)
        self.combo_slice_merge_metric.setEnabled(False)
        self.combo_slice_fragment_policy.setEnabled(False)
        self._load_spec(self._base, {})

    # ------------------------------------------------------------ public API

    @property
    def role(self) -> str:
        return self._role

    @property
    def capabilities(self) -> SliceWidgetCapabilities:
        return self._caps

    def set_spec(self, spec: TilingSpec, *, extras: dict | None = None) -> None:
        """Show ``spec`` (+ role extras). The ONLY place the widget writes controls."""
        self._load_spec(spec, dict(extras or {}))
        self._refresh()

    def spec(self) -> TilingSpec:
        role = self._role
        fractions = self._fractions()
        values: dict[str, Any] = {
            "reference_body_px": self._body_value(),
            "slice_width": int(self.spin_slice_tile_w.value()),
            "slice_height": int(self.spin_slice_tile_h.value()),
            "overlap": min(float(self.spin_slice_overlap.value()), OVERLAP_MAX),
            "object_tile_fractions": tuple(fractions),
        }
        if "enabled" in self._role_rows:
            values["enabled"] = self.chk_slice_enabled.isChecked()
        elif role == "train_sam3":
            values["enabled"] = True  # SAM3 training always tiles
        else:
            values["enabled"] = bool(fractions)
        if "mode" in self._role_rows:
            values["geometry_mode"] = self._mode()
        if "min_area" in self._role_rows:
            values["min_area_ratio"] = float(self.spin_slice_min_area.value())
        if "merge" in self._role_rows and role != "escalate_sam3":
            values["merge_threshold"] = float(self.spin_slice_merge.value())
        if "merge_policy" in self._role_rows:
            values["merge_policy"] = str(self.combo_slice_merge_policy.currentData())
            values["merge_metric"] = str(self.combo_slice_merge_metric.currentData())
        return replace(self._base, **values)

    def extras(self) -> dict:
        out = dict(self._passthrough)
        role = self._role
        if role == "train_yolo":
            out.update(
                negative_tile_fraction=float(self.spin_slice_negative.value()),
                full_frame_mix=self.chk_slice_full_frame_mix.isChecked(),
                balance_multiscale_loss=self.chk_slice_balance_loss.isChecked(),
                balance_multiscale_loss_power=float(
                    self.spin_slice_balance_power.value()
                ),
                min_area_ratio=float(self.spin_slice_min_area.value()),
            )
        elif role == "train_sam3":
            out.update(
                object_tile_fraction=float(self.spin_slice_object_fraction.value()),
                keep_empty_tiles=self.chk_slice_keep_empty.isChecked(),
                full_frame_mix=self.chk_slice_full_frame_mix.isChecked(),
                min_area_ratio=float(self.spin_slice_min_area.value()),
                tile_overlap=float(self.spin_slice_overlap.value()),
            )
        elif role == "infer_yolo":
            if "merge" in self._role_rows:
                out["merge_threshold"] = float(self.spin_slice_merge.value())
            if "full_pass" in self._role_rows:
                out["perform_standard_pred"] = (
                    self.chk_slice_full_frame_pass.isChecked()
                )
            if "tile_batch" in self._role_rows:
                out["tile_batch_size"] = int(self.spin_slice_tile_batch.value())
                out["memory_budget_mib"] = int(self.spin_slice_memory_budget.value())
        elif role == "escalate_sam3":
            out.update(
                seam_margin_px=int(self.spin_slice_seam_margin.value()),
                merge_iou=float(self.spin_slice_merge.value()),
            )
        return out

    def set_model_input_size(self, imgsz: int) -> None:
        """The real model input; only used to DISPLAY px (never a literal 640)."""
        self._model_input_size = max(1, int(imgsz))
        self._refresh_scales_tooltip()
        self._refresh()

    def model_input_size(self) -> int:
        return self._model_input_size

    def set_preview_frame_size(self, frame_wh: tuple[int, int] | None) -> None:
        self.preview.set_frame_size(frame_wh)

    def set_preview_frame_options(self, options: list[tuple[int, int, int]]) -> None:
        self.preview.set_frame_options(options)

    def set_reference_body(self, value: float, source: str) -> None:
        """Show a derived body size and its source; read-only until Override."""
        value = max(0.0, float(value or 0.0))
        source = source if value > 0 else "default"
        self._body_derived = (value, source)
        self._body_source = source
        self._set_quietly(self.chk_slice_body_override, False)
        self._set_quietly(self.spin_slice_body, value)
        self._refresh()

    def set_source(self, field: str, source: str) -> None:
        """Badge a field's source (e.g. fractions ``stamped``/``profile``)."""
        self._sources[field] = source
        self._refresh()

    def source_badge(self, field: str) -> str:
        if field == "reference_body_px":
            return self._body_source
        if field == "tile_size":
            return "user" if self._mode() == "custom" else "derived"
        return self._sources.get(field, "user")

    def set_advanced_expanded(self, expanded: bool) -> None:
        self._advanced_expanded = bool(expanded)
        self._set_quietly(self.btn_slice_advanced, self._advanced_expanded)
        self.btn_slice_advanced.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._apply_visibility()

    def set_profile_row_visible(self, visible: bool) -> None:
        """The profile row appears only when the host has a model sidecar."""
        self._profile_row_shown = bool(visible)
        self._apply_visibility()

    # -------------------------------------------------------------- internals

    def _mode(self) -> str:
        if "mode" in self._role_rows:
            return str(self.combo_slice_geometry.currentData() or "auto_object")
        return self._base.geometry_mode

    def _parsed_scales(self) -> list[float]:
        fractions: list[float] = []
        for token in self.txt_slice_scales.text().split(","):
            try:
                value = float(token.strip())
            except ValueError:
                continue
            if 0.0 < value <= 1.0:
                fractions.append(value)
        return fractions

    def _fractions(self) -> list[float]:
        if self._role in TRAIN_ROLES:
            return self._parsed_scales()
        value = float(self.spin_slice_object_fraction.value())
        return [value] if value > 0.0 else []

    def _display_fractions(self) -> list[float]:
        """Scales the preview/suggestion use (fallbacks a host would apply)."""
        fractions = self._fractions()
        if fractions:
            return fractions
        if self._role == "train_yolo":
            return list(BACKEND_DEFAULTS["yolo_train"].object_tile_fractions)
        if self._role == "train_sam3":
            return [float(self.spin_slice_object_fraction.value())]
        return []

    def _body_value(self) -> float:
        if self._role in TRAIN_ROLES:
            return float(self._body_derived[0])
        return float(self.spin_slice_body.value())

    @staticmethod
    def _set_quietly(widget: QWidget, value: Any) -> None:
        widget.blockSignals(True)
        try:
            if isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, (QCheckBox, QToolButton)):
                widget.setChecked(bool(value))
            elif hasattr(widget, "findData"):
                index = widget.findData(value)
                widget.setCurrentIndex(index if index >= 0 else 0)
            else:
                widget.setValue(value)
        finally:
            widget.blockSignals(False)

    def _load_spec(self, spec: TilingSpec, extras: dict) -> None:
        role = self._role
        q = self._set_quietly
        self._base = spec
        q(self.chk_slice_enabled, spec.enabled)
        q(self.combo_slice_geometry, spec.geometry_mode)
        fractions = list(spec.object_tile_fractions)
        q(self.txt_slice_scales, ", ".join(f"{value:.8g}" for value in fractions))
        if role == "infer_yolo":
            fraction = spec.operating_fraction()
            if fraction is not None:
                q(self.spin_slice_object_fraction, fraction)
        elif role in ESCALATE_ROLES:
            q(
                self.spin_slice_object_fraction,
                fractions[0] if spec.enabled and fractions else 0.0,
            )
        body = float(spec.reference_body_px)
        if role in TRAIN_ROLES:
            source = "dataset" if body > 0 else "default"
        else:
            source = "user" if body > 0 else "default"
        self._body_derived = (body, source)
        self._body_source = source
        q(self.chk_slice_body_override, False)
        q(self.spin_slice_body, body)
        q(self.spin_slice_tile_w, spec.slice_width)
        q(self.spin_slice_tile_h, spec.slice_height)
        if self._caps.fixed_overlap is None and spec.overlap is not None:
            q(self.spin_slice_overlap, spec.overlap)
        q(self.combo_slice_merge_policy, spec.merge_policy)
        q(self.combo_slice_merge_metric, spec.merge_metric)
        q(self.combo_slice_fragment_policy, spec.fragment_policy)
        q(self.spin_slice_min_area, spec.min_area_ratio)
        if role != "escalate_sam3":
            q(self.spin_slice_merge, spec.merge_threshold)
        self._load_extras(extras)

    _ROLE_EXTRAS = {
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

    def _load_extras(self, extras: dict) -> None:
        owned = self._ROLE_EXTRAS[self._role]
        self._passthrough = {k: v for k, v in extras.items() if k not in owned}
        for key, attr in owned.items():
            if key in extras and extras[key] is not None:
                widget = getattr(self, attr)
                value = extras[key]
                if isinstance(widget, QCheckBox):
                    self._set_quietly(widget, bool(value))
                elif attr in ("spin_slice_tile_batch", "spin_slice_memory_budget"):
                    self._set_quietly(widget, int(value))
                elif attr == "spin_slice_seam_margin":
                    self._set_quietly(widget, int(round(float(value))))
                else:
                    self._set_quietly(widget, float(value))

    def _on_user_edit(self, field: str) -> None:
        if field == "reference_body_px":
            self._body_source = (
                "override" if self.chk_slice_body_override.isChecked() else "user"
            )
        elif field == "body_override":
            if self.chk_slice_body_override.isChecked():
                self._body_source = "override"
            else:
                value, source = self._body_derived
                self._body_source = source
                self._set_quietly(self.spin_slice_body, value)
        elif field == "merge_policy":
            self._passthrough.pop("merge_policy_raw", None)
        self._refresh()
        self.field_changed.emit(field)

    def _use_suggested(self) -> None:
        suggested = self._suggested_overlap()
        self.spin_slice_overlap.setValue(suggested)  # the user path: emits

    def _suggested_overlap(self) -> float:
        fractions = self._display_fractions()
        return float(resolve_overlap(fractions=fractions).value)

    # ------------------------------------------------------------- refreshes

    def _refresh(self) -> None:
        self._refresh_enabled()
        self._refresh_derived()
        self._apply_visibility()

    def _body_editable(self) -> bool:
        if self.chk_slice_body_override.isChecked():
            return True
        return self._body_source in ("user", "default") and (
            self._caps.body_override or self._body_source == "default"
        )

    def _refresh_enabled(self) -> None:
        role = self._role
        on = (
            self.chk_slice_enabled.isChecked() if "enabled" in self._role_rows else True
        )
        mode = self._mode()
        auto_object = mode == "auto_object"
        fixed = self._caps.fixed_overlap is not None
        derived_body = self._body_derived[1] not in ("user", "default")
        body_gate = on and (role != "infer_yolo" or auto_object)
        enabled = {
            self.combo_slice_profile: on,
            self.combo_slice_geometry: on,
            self.txt_slice_scales: on and auto_object,
            self.spin_slice_object_fraction: on
            and (role != "infer_yolo" or auto_object),
            self.spin_slice_body: body_gate and self._body_editable(),
            self.chk_slice_body_override: body_gate
            and self._caps.body_override
            and derived_body,
            self.spin_slice_tile_w: on and mode == "custom",
            self.spin_slice_tile_h: on and mode == "custom",
            self.spin_slice_overlap: on and not fixed,
            self.btn_slice_overlap_use_suggested: on
            and not fixed
            and abs(self.spin_slice_overlap.value() - self._suggested_overlap())
            > 10 ** -(self.spin_slice_overlap.decimals() + 1),
            self.spin_slice_merge: on,
            self.spin_slice_seam_margin: on,
            self.spin_slice_min_area: on,
            self.spin_slice_negative: on,
            self.chk_slice_keep_empty: on,
            self.chk_slice_full_frame_mix: on,
            self.chk_slice_full_frame_pass: on,
            self.chk_slice_balance_loss: on and auto_object,
            self.spin_slice_balance_power: on and auto_object,
            self.spin_slice_tile_batch: on,
            self.spin_slice_memory_budget: on,
        }
        for widget, state in enabled.items():
            widget.setEnabled(bool(state))
        body_tip = (
            "The typical longest side of one animal, in source-image pixels. "
            "Tile size = this / object scale."
        )
        if not self.spin_slice_body.isEnabled() and derived_body:
            where = SOURCE_DESCRIPTIONS.get(self._body_derived[1], "")
            body_tip += f" Derived from {where}; check Override to edit."
        self.spin_slice_body.setToolTip(body_tip)

    def _refresh_derived(self) -> None:
        role = self._role
        fractions = self._display_fractions()
        body = self._body_value()
        mode = self._mode()
        imgsz = self._model_input_size
        if role == "infer_yolo":
            px = float(self.spin_slice_object_fraction.value()) * imgsz
            self.lbl_slice_scale_px.setText(f"≈ {px:.1f} px at {imgsz} px input")
        if role in ESCALATE_ROLES:
            fraction = fractions[0] if fractions else None
            tile = None
            if fraction is not None and body > 0:
                tile = tile_size_for_mode(
                    geometry_mode="auto_object",
                    imgsz=imgsz,
                    reference_body_px=body,
                    object_tile_fraction=fraction,
                    slice_width=0,
                    slice_height=0,
                )[0]
            formatter = self._caps.tile_label_formatter or default_tile_label
            text, tip = formatter(tile, body, fraction)
            self.lbl_slice_tile_size.setText(text)
            self.lbl_slice_tile_size.setToolTip(tip)
        else:
            self.lbl_slice_tile_size.setText(self._tile_text(mode, fractions, body))
        set_badge(self.lbl_slice_tile_badge, self.source_badge("tile_size"))
        set_badge(self.lbl_slice_body_badge, self._body_source)
        if role in TRAIN_ROLES:
            self._refresh_reference_note(body)
        if self._caps.fixed_overlap is None:
            suggested = self._suggested_overlap()
            self.lbl_slice_overlap_suggested.setText(f"Suggested: {suggested:.2f}")
            self.lbl_slice_overlap_suggested.setToolTip(
                "Largest object scale + margin: every animal fits whole inside at "
                "least one tile. A suggestion only — your overlap is kept."
            )
        if role not in ESCALATE_ROLES:
            self.preview.set_settings(
                mode=mode,
                target_fractions=fractions
                or [BACKEND_DEFAULTS["yolo_infer"].object_tile_fractions[0]],
                slice_width=self.spin_slice_tile_w.value(),
                slice_height=self.spin_slice_tile_h.value(),
                overlap=self.spin_slice_overlap.value(),
                model_input_size=imgsz,
                reference_body_px=body,
            )

    def _tile_text(self, mode: str, fractions: list[float], body: float) -> str:
        imgsz = self._model_input_size
        if mode == "custom":
            w = self.spin_slice_tile_w.value() or imgsz
            h = self.spin_slice_tile_h.value() or imgsz
            return f"→ {w} × {h} px"
        if mode == "auto_model" or not fractions:
            return f"→ {imgsz} × {imgsz} px (model input)"
        if body <= 0:
            if self._role in TRAIN_ROLES:
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

    def _refresh_reference_note(self, body: float) -> None:
        if body > 0.0:
            text = f"Last build: {body:.1f} px, measured automatically from labels."
        else:
            text = (
                "Measured automatically from all labelled objects when the "
                "sliced dataset is built."
            )
        self.auto_reference_note.setText(text)

    def _refresh_scales_tooltip(self) -> None:
        size = self._model_input_size
        self.txt_slice_scales.setToolTip(
            f"Object size as a fraction of the model input. At a {size}px input, "
            f"0.31 means about {0.31 * size:.0f}px. Larger fractions create "
            "smaller tiles. Editable with 'Fit to animal size'."
        )
        row = getattr(self, "_rows", {}).get("targets")
        if row is not None:
            row[0].setToolTip(self.txt_slice_scales.toolTip())

    def _apply_visibility(self) -> None:
        mode = self._mode()
        for key, widgets in self._row_widgets.items():
            visible = key in self._role_rows
            if key in _ADVANCED:
                visible = visible and self._advanced_expanded
            if key == "profile":
                visible = visible and self._profile_row_shown
            if key == "body" and self._role in TRAIN_ROLES:
                # A note, not a field: meaningful only when fitting to animals.
                visible = visible and mode == "auto_object"
            for widget in widgets:
                widget.setHidden(not visible)
