"""One shared SAHI settings widget for every host (spec §5, S4 Task 15).

The widget's state is the S1 contract (:class:`TilingSpec`) plus a
role-specific ``extras`` dict. Ownership model (plan decision 23): the widget
writes its own controls ONLY inside :meth:`SliceSettingsWidget.set_spec`
(signals blocked, then derived labels refreshed). Derived values (tile size
outside Custom, the whole-animal overlap minimum, the resolved tile) are shown in
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
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.utils.slice_geometry import tile_size_for_mode
from hydra_suite.utils.tiling_spec import (
    BACKEND_DEFAULTS,
    DEFAULT_YOLO_IMGSZ,
    OVERLAP_MAX,
    TilingSpec,
)

from .slice_settings_controls import (
    ADVANCED,
    FULL_WIDTH,
    ROLE_EXTRAS,
    build_controls,
    refresh_overlap_minimum,
    role_keys,
    row_specs,
)
from .slice_settings_parts import (
    ESCALATE_ROLES,
    PREVIEW_POSITIONS,
    ROLE_BACKEND,
    ROLES,
    SOURCE_DESCRIPTIONS,
    TRAIN_ROLES,
    SliceWidgetCapabilities,
    default_capabilities,
    default_tile_label,
    set_badge,
    tile_text,
    whole_animal_minimum,
    widget_stylesheet,
)

__all__ = ["ROLES", "SliceSettingsWidget", "SliceWidgetCapabilities"]

# Advanced rows, collapsed by default.


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
        if self._caps.preview_position not in PREVIEW_POSITIONS:
            raise ValueError(
                f"unknown preview position: {self._caps.preview_position!r}"
            )
        self._base = TilingSpec.defaults(ROLE_BACKEND[role])
        self._passthrough: dict[str, Any] = {}
        self._model_input_size = DEFAULT_YOLO_IMGSZ
        self._body_derived = (0.0, "user")
        self._body_source = "user"
        self._sources: dict[str, str] = {}
        self._source_notes: dict[str, str] = {}
        self._advanced_expanded = False
        self._profile_row_shown = False
        self._loading = False
        # Constrained controls must LOOK disabled under any host theme (the
        # DetectKit dark theme styles inputs but not their :disabled state).
        self._bare = not title
        if self._bare:
            self.setFlat(True)
        self._apply_style(None)

        build_controls(self)
        self._build_layout(bare=not title)
        self._wire()
        self._apply_role_defaults()
        self._refresh()

    # ------------------------------------------------------------------ style

    def _apply_style(self, text_color: str | None) -> None:
        """Widget-scoped styling that reads correctly under any host theme.

        Constrained controls must LOOK disabled even where the host theme
        styles inputs but not their ``:disabled`` state (DetectKit's dark
        theme). Tool buttons take the host's label colour, which a theme sets
        through a stylesheet the buttons themselves do not match.
        """
        style = widget_stylesheet(text_color, bare=self._bare)
        self.setStyleSheet(style)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        probe = self._rows.get("mode", self._rows.get("body"))
        if probe is None:
            return
        color = probe[0].palette().color(probe[0].foregroundRole()).name()
        if color != getattr(self, "_styled_text_color", None):
            self._styled_text_color = color
            self._apply_style(color)

    # ------------------------------------------------------------------ build

    def _build_layout(self, *, bare: bool) -> None:
        bottom = self._caps.preview_position == "bottom"
        outer = QVBoxLayout(self) if bottom else QHBoxLayout(self)
        if bare:
            outer.setContentsMargins(0, 0, 0, 0)
        else:
            outer.setContentsMargins(14, 16, 14, 12)
        outer.setSpacing(10 if bottom else 18)
        controls = QWidget()
        self._controls = controls
        grid = QGridLayout(controls)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(7)
        grid.setColumnStretch(1, 1)
        self._grid = grid
        self._role_rows = role_keys(self)
        # Legacy name kept for DetectKit tests: key -> (label, control).
        self._rows: dict[str, tuple[QLabel, QWidget]] = {}
        self._row_widgets: dict[str, list[QWidget]] = {}
        for row, (key, text, control, badge) in enumerate(row_specs(self)):
            widgets: list[QWidget] = [control]
            if text is None or key in FULL_WIDTH or key == "advanced":
                grid.addWidget(control, row, 0, 1, 3)
            else:
                label = QLabel(text)
                label.setToolTip(self._label_tooltip(control))
                if key in self._stacked_rows:
                    # Level with the control line, not centred on the note.
                    top = control.layout().itemAt(0).widget()
                    label.setMinimumHeight(top.sizeHint().height())
                    grid.addWidget(label, row, 0, Qt.AlignmentFlag.AlignTop)
                else:
                    grid.addWidget(label, row, 0)
                grid.addWidget(control, row, 1)
                self._rows[key] = (label, control)
                widgets.append(label)
            if badge is not None:
                if key in self._stacked_rows:
                    badge.setMinimumHeight(label.minimumHeight())
                    grid.addWidget(badge, row, 2, Qt.AlignmentFlag.AlignTop)
                else:
                    grid.addWidget(badge, row, 2)
                widgets.append(badge)
            self._row_widgets[key] = widgets
        grid.setRowStretch(len(self._row_widgets), 1)
        outer.addWidget(controls, 0)
        if self._role in ESCALATE_ROLES:
            self.preview.hide()
        else:
            outer.addWidget(self.preview, 0 if bottom else 1)

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
        self.btn_slice_overlap_raise.clicked.connect(self._raise_to_minimum)
        self.btn_slice_advanced.toggled.connect(self.set_advanced_expanded)

    def _apply_role_defaults(self) -> None:
        caps = self._caps
        if caps.fixed_overlap is not None:
            self.spin_slice_overlap.setValue(float(caps.fixed_overlap))
            self.spin_slice_overlap.setToolTip(
                "Fixed: owner tiles always overlap by this fraction."
            )
        self.btn_slice_overlap_raise.setVisible(False)
        if caps.fixed_overlap is not None:
            self.lbl_slice_overlap_minimum.setText("fixed")
            self.lbl_slice_overlap_minimum.setToolTip(
                "This backend's tiles always overlap by this constant."
            )
        self._tile_spins.setVisible(self._role not in ESCALATE_ROLES)
        self.lbl_slice_scale_px.setVisible(self._role == "infer_yolo")
        # Merge policy/metric are display-only here: no S4a host persists them
        # (DetectKit reads them from the model's profile).
        self.combo_slice_merge_policy.setEnabled(False)
        self.combo_slice_merge_metric.setEnabled(False)
        self.combo_slice_fragment_policy.setEnabled(False)
        base, extras = self._base, {}
        if self._role == "train_sam3":
            # SAM3's scale SET defaults to empty (= use the scalar), so the
            # defaults-table scale seeds the scalar, never the set.
            extras = {"object_tile_fraction": base.object_tile_fractions[0]}
            base = replace(base, object_tile_fractions=())
        self._load_spec(base, extras)

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
        # An unknown body is the user's to enter: badged "user" (unless the
        # host only displays a body it owns elsewhere).
        if value <= 0 and not self._caps.body_display_only:
            source = "user"
        self._body_derived = (value, source)
        self._body_source = source
        self._set_quietly(self.chk_slice_body_override, False)
        self._set_quietly(self.spin_slice_body, value)
        self._refresh()

    def refresh(self) -> None:
        """Recompute derived labels, badges and enablement (no control writes)."""
        self._refresh()

    def set_source(self, field: str, source: str, *, note: str = "") -> None:
        """Badge a field's source (e.g. fractions ``stamped``/``profile``).

        ``"tile_size"`` badges a custom tile size the host applied (shown only
        in Custom; any user edit of W/H or the mode resets it to ``user``).
        ``"overlap"`` from a profile/stamp is never nudged: the whole-animal
        minimum shows as muted info naming ``note`` (else the source's
        description), with no Raise, until the user edits the overlap.
        """
        self._sources[field] = source
        self._source_notes[field] = note
        self._refresh()

    def source_badge(self, field: str) -> str:
        if field == "reference_body_px":
            return self._body_source
        if field == "tile_size":
            if self._mode() != "custom":
                return "derived"
            # A host-applied custom size (a profile, a stamp) keeps its
            # source until the user edits the size or the mode.
            return self._sources.get("tile_size", "user")
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
        self._refresh()

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
        """Scales the preview and overlap minimum use (host fallbacks applied)."""
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
            # Always measured from the labels at build time (0 = not yet).
            source = "dataset"
        else:
            source = "user"
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

    def _load_extras(self, extras: dict) -> None:
        owned = ROLE_EXTRAS[self._role]
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
        elif field in ("slice_width", "slice_height", "geometry_mode"):
            self._sources.pop("tile_size", None)
            # A profile/stamp overlap was measured for ITS geometry; once the
            # user changes the geometry the claim no longer applies.
            self._sources.pop("overlap", None)
        elif field in ("object_tile_fractions", "object_tile_fraction"):
            self._sources.pop("overlap", None)
        elif field == "overlap":
            self._sources.pop("overlap", None)
        elif field == "merge_policy":
            self._passthrough.pop("merge_policy_raw", None)
        self._refresh()
        self.field_changed.emit(field)

    def _raise_to_minimum(self) -> None:
        minimum = self._whole_animal_minimum()
        if minimum is not None:
            self.spin_slice_overlap.setValue(minimum[0])  # the user path: emits

    def _minimum_fractions(self) -> list[float]:
        """The animal's share of a tile, per tile size the mode produces.

        auto_object: the object scale(s) themselves (tile = body / scale).
        custom: body / min(W, H) (a 0 side is the model input); auto_model:
        body / model input. Outside auto_object the object scale is unused,
        so without a known body there is no minimum to state.
        """
        mode = self._mode()
        if mode == "auto_object":
            return self._display_fractions()
        body = self._body_value()
        if body <= 0:
            return []
        imgsz = self._model_input_size
        if mode == "custom":
            side = min(
                int(self.spin_slice_tile_w.value()) or imgsz,
                int(self.spin_slice_tile_h.value()) or imgsz,
            )
        else:
            side = imgsz
        return [body / float(side)] if side > 0 else []

    def _whole_animal_minimum(self) -> tuple[float, bool] | None:
        return whole_animal_minimum(
            self._minimum_fractions(),
            decimals=self.spin_slice_overlap.decimals(),
            ceiling=self.spin_slice_overlap.maximum(),
        )

    # ------------------------------------------------------------- refreshes

    def _refresh(self) -> None:
        self._refresh_enabled()
        self._refresh_derived()
        self._apply_visibility()

    def _body_editable(self) -> bool:
        if self._caps.body_display_only:
            return False
        value, _source = self._body_derived
        if self.chk_slice_body_override.isChecked() or value <= 0:
            return True  # I6: an unknown body always stays typeable
        return self._body_source == "user" and self._caps.body_override

    def _body_is_derived(self) -> bool:
        value, source = self._body_derived
        return value > 0 and source not in ("user", "default")

    def _refresh_enabled(self) -> None:
        role = self._role
        on = (
            self.chk_slice_enabled.isChecked() if "enabled" in self._role_rows else True
        )
        mode = self._mode()
        auto_object = mode == "auto_object"
        fixed = self._caps.fixed_overlap is not None
        derived_body = self._body_is_derived()
        body_gate = on and (role != "infer_yolo" or auto_object)
        enabled = {
            # Profiles own `enabled`: picking one from a SAHI-off state
            # applies it and turns SAHI on, so the picker is never gated on
            # the checkbox (only on the row being shown).
            self.combo_slice_profile: self._profile_row_shown,
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
            self.btn_slice_overlap_raise: on and not fixed,
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
        # Override exists only for a DERIVED body (never for a user's own).
        self.chk_slice_body_override.setHidden(
            not (
                self._caps.body_override
                and role not in TRAIN_ROLES
                and (derived_body or self.chk_slice_body_override.isChecked())
            )
        )

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
            self.lbl_slice_tile_size.setText(
                tile_text(
                    mode=mode,
                    fractions=fractions,
                    body=body,
                    imgsz=self._model_input_size,
                    custom_wh=(
                        self.spin_slice_tile_w.value(),
                        self.spin_slice_tile_h.value(),
                    ),
                    training=self._role in TRAIN_ROLES,
                )
            )
        set_badge(self.lbl_slice_tile_badge, self.source_badge("tile_size"))
        if body > 0:
            set_badge(self.lbl_slice_body_badge, self._body_source)
        else:
            # Unknown (0): there is no source to claim.
            self.lbl_slice_body_badge.setText("")
            self.lbl_slice_body_badge.setToolTip("No body size is known yet.")
        if role in TRAIN_ROLES:
            self._refresh_reference_note(body)
        if self._caps.fixed_overlap is None:
            refresh_overlap_minimum(self)
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
            if key in ADVANCED:
                visible = visible and self._advanced_expanded
            if key == "profile":
                visible = visible and self._profile_row_shown
            if key == "body" and self._role in TRAIN_ROLES:
                # A note, not a field: meaningful only when fitting to animals.
                visible = visible and mode == "auto_object"
            for widget in widgets:
                widget.setHidden(not visible)
        self.updateGeometry()  # let host layouts re-measure (rows came/went)
