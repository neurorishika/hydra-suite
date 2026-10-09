"""SemanticEscalationDialog — prompt, tiling, calibration, visual test frame."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.core.inference.geometry_drift import (
    compare_geometry_value,
    stamped_object_tile_fraction,
)
from hydra_suite.core.inference.semantic.calibration_record import (
    CalibrationOrigin,
    resolve_serving_calibration,
    write_serving_calibration,
)
from hydra_suite.core.inference.semantic.checkpoints import (
    CHECKPOINT_SIZE_GB,
    available_models,
    available_variants,
    probe_checkpoint,
    sidecar_for,
    sidecar_path_for,
)
from hydra_suite.core.inference.semantic.tiling import (
    DEFAULT_MERGE_IOU,
    DEFAULT_OVERLAP,
    DEFAULT_SEAM_MARGIN_PX,
    SEMANTIC_TILE_FRACTION_SEED,
)
from hydra_suite.detectkit.gui.dialogs._base import DetectKitDialog
from hydra_suite.detectkit.gui.widgets.calibration_source_selector import (
    NO_POLYGON_SOURCES,
    CalibrationSourceSelector,
    scale_warning_text,
)
from hydra_suite.utils.hidden_files import is_hidden_file
from hydra_suite.utils.tiling_spec import FRACTION_MAX, OVERLAP_MAX, TilingSpec
from hydra_suite.widgets.device_combo import DeviceCombo
from hydra_suite.widgets.slice_settings import (
    SliceSettingsWidget,
    SliceWidgetCapabilities,
)


def _saved_value(saved: dict, key: str, default, cast):
    """Read a hand-editable persisted setting without making the dialog fragile."""
    try:
        return cast(saved.get(key, default))
    except (TypeError, ValueError):
        return default


def _body_source(explicit: str, origin: str) -> str:
    """The body-size badge: the caller's source, else read from its note."""
    if explicit:
        return explicit
    text = (origin or "").lower()
    if "sliced-training" in text:
        return "project"
    if "median" in text:
        return "dataset"
    return "user"


def _format_tile_label(
    tile_px: int | None, body_px: float, fraction: float | None
) -> tuple[str, str]:
    """This dialog's resolved-tile wording (the shared widget renders it)."""
    if tile_px:
        return (
            f"{tile_px} px\n{body_px:.0f} px / {fraction:.2f}",
            f"Tile size {tile_px} px = {body_px:.0f} px reference body "
            f"size / {fraction:.2f} tile fraction.",
        )
    if fraction is None:
        return "full frame — tiling off by choice.", ""
    # Short visible line (no wrapping, so the row never squeezes its
    # neighbours); the guidance lives in the tooltip.
    return (
        "full frame — tiling is off; enter a body size",
        "Full frame: no reference body size is known, so tiling is off. Enter "
        "one above (or set one in project settings) for much better "
        "small-object recall.",
    )


class SemanticEscalationDialog(DetectKitDialog):
    """Configure a SAM3 semantic escalation run.

    Calibration runs on the "Calibrate on" list: sources with frames
    labelled entirely in polygons. It scores masks, so it needs real ground
    truth -- a box is not one. Escalation runs on the separate "Escalate"
    list, which is usually the sources WITHOUT that ground truth.
    """

    def __init__(
        self,
        sources,
        reference_body_px: float,
        parent=None,
        body_px_origin: str = "",
        project=None,
        persist_callback=None,
        body_px_source: str = "",
    ) -> None:
        super().__init__(
            "Semantic escalation (SAM3)",
            parent=parent,
            buttons=(
                QDialogButtonBox.StandardButton.Ok
                | QDialogButtonBox.StandardButton.Cancel
            ),
        )
        self._sources = list(sources)
        self._body_px_origin = body_px_origin
        self._project = project
        self._persist_callback = persist_callback
        saved = dict(getattr(project, "semantic_escalation_settings", {}) or {})
        # Restored, not rebuilt: refitting the band would need the labelled
        # frames re-read, and a reopened dialog must offer the same gate the
        # last calibration chose.
        self._area_band = (
            _saved_value(saved, "area_min_px2", 0.0, float),
            _saved_value(saved, "area_max_px2", 0.0, float),
        )
        # D10: a calibration now lives on the MODEL SIDECAR, which is what a
        # headless serving run can see. The project's own copy is still
        # written and is still read here as the LEGACY path, so a project
        # calibrated before this change keeps working -- it is reported as
        # legacy-only, never silently migrated onto the sidecar (that would
        # rewrite a user's artifacts behind their back).
        record, self._calibration_origin = resolve_serving_calibration(
            sidecar_for(str(saved.get("variant", ""))),
            getattr(project, "semantic_calibration", {}) or {},
        )
        self._saved_calibration = dict(record or {})
        self.calibration_points = self._restore_calibration_points(
            self._saved_calibration
        )
        self.calibration_preview_frames: list = []
        self._preview_worker = None

        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(12)

        self._selector = CalibrationSourceSelector(self._sources)
        self._selector.restore(
            saved.get("calibration_source_paths"),
            saved.get("source_names"),
            saved.get("escalation_source_paths"),
        )
        # The escalation list keeps its old name: handlers and tests drive it.
        self._list = self._selector.escalation_list
        top.addWidget(self._selector, 2)

        settings_group = QGroupBox("Run settings")
        form = QGridLayout(settings_group)
        self._settings_grid = form
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(7)
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)

        def add_field(row: int, pair: int, label: str, widget: QWidget) -> None:
            column = pair * 2
            form.addWidget(QLabel(label), row, column)
            form.addWidget(widget, row, column + 1)

        self._variant = QComboBox()
        self._variant.addItems(available_models())
        saved_variant = str(saved.get("variant", ""))
        if self._variant.findText(saved_variant) >= 0:
            self._variant.setCurrentText(saved_variant)
        add_field(0, 0, "Model", self._variant)

        # The dropdown is populated from available_models() = stock variants
        # + registry-published finetuned models. Until a finetuned model is
        # published that is exactly ["sam3"] -- one item, no explanation --
        # which reads as "model selection was never built" rather than
        # "nothing to select yet". Say so explicitly.
        self._no_finetuned_hint = QLabel(
            "No finetuned SAM3 models published yet — train one from the "
            "DetectKit training dialog (Semantic mode)."
        )
        self._no_finetuned_hint.setWordWrap(True)
        finetuned_present = len(available_models()) > len(available_variants())
        self._no_finetuned_hint.setVisible(not finetuned_present)
        form.addWidget(self._no_finetuned_hint, 5, 0, 1, 4)

        self._prompt = QLineEdit(str(saved.get("prompt", "ant") or "ant"))
        self._prompt.setToolTip(
            "A noun phrase. Wording matters far less than tile size — try "
            "variants in the preview if results look wrong."
        )
        add_field(0, 1, "Prompt", self._prompt)

        # The prompt is a noun phrase tuned to make the MODEL find things;
        # the class is what the found things ARE in this project. Conflating
        # them wrote the prompt into the staging dir's classes.txt, so
        # accepting the review appended a class the project had never heard
        # of -- and the overlay and the training dataset builder both drop
        # labels outside the project scheme, leaving the accepted work blank
        # on canvas and absent from training.
        self._class_name = QComboBox()
        project_classes = [
            str(name)
            for name in (getattr(self._project, "class_names", None) or [])
            if str(name).strip()
        ]
        self._class_name.addItems(project_classes)
        saved_class = str(saved.get("class_name", "") or "")
        if self._class_name.findText(saved_class) >= 0:
            self._class_name.setCurrentText(saved_class)
        # A project with no class list is not a state this dialog can
        # invent a class for: disable rather than silently pick one, and
        # `class_name()` then returns "" so the job falls back to the
        # prompt -- the pre-fix behaviour, which at least stages something.
        self._class_name.setEnabled(bool(project_classes))
        self._class_name.setToolTip(
            "The project class the staged instances will be labelled as. "
            "This is what accept writes into the source -- not the prompt."
        )
        add_field(3, 0, "Assign to class", self._class_name)

        self._device = DeviceCombo(str(saved.get("device", "auto") or "auto"))
        add_field(3, 1, "Run on", self._device)

        self._confidence = QDoubleSpinBox()
        self._confidence.setRange(0.01, 0.99)
        self._confidence.setSingleStep(0.05)
        self._confidence.setValue(_saved_value(saved, "confidence", 0.35, float))
        add_field(1, 0, "Confidence", self._confidence)

        self._max_instances = QSpinBox()
        self._max_instances.setRange(0, 10000)
        self._max_instances.setSpecialValueText("unlimited")
        self._max_instances.setValue(_saved_value(saved, "max_instances", 0, int))
        add_field(1, 1, "Max instances/tile", self._max_instances)

        # Tiling rows come from the shared SAHI widget (S4). The old attribute
        # names alias its controls, so parameters(), persistence, calibration
        # and prefill keep working unchanged.
        self._tiling = SliceSettingsWidget(
            role="escalate_sam3",
            title="Tiling (SAHI)",
            capabilities=SliceWidgetCapabilities(
                advanced_merge=False, tile_label_formatter=_format_tile_label
            ),
        )
        tiling = self._tiling
        self._overlap = tiling.spin_slice_overlap
        self._seam_margin = tiling.spin_slice_seam_margin
        self._merge_iou = tiling.spin_slice_merge
        # I6: link 3 of the reference_body_px resolution chain (project
        # setting -> median of the source's existing labels -> THE USER).
        # An unknown (0) body stays typeable without Override; a derived one
        # is read-only until Override, with its source badged.
        self._reference_body = tiling.spin_slice_body
        # The fraction is a CALIBRATED parameter. The seed is presented as a
        # guess, never as a tuned or recommended value -- it was back-derived
        # from one measured configuration on one dataset. 3 decimals: a
        # published model's sidecar can record e.g. 0.055.
        self._tile_fraction = tiling.spin_slice_object_fraction
        self._tile_label = tiling.lbl_slice_tile_size

        body = _saved_value(
            saved, "reference_body_px", float(reference_body_px or 0.0), float
        )
        fraction = _saved_value(
            saved, "tile_fraction", SEMANTIC_TILE_FRACTION_SEED, float
        )
        if "reference_body_px" in saved:
            origin_text, body_source = "saved from the previous SAM3 dialog", "user"
        else:
            origin_text = body_px_origin or "entered by you"
            body_source = _body_source(body_px_source, body_px_origin)
        self._body_origin_label = QLabel(origin_text)
        self._body_origin_label.setWordWrap(True)
        self._body_origin_label.setToolTip(self._body_origin_label.text())

        # F3: when nothing saved applies to the opening variant, open where a
        # headless `detectkit escalate sam3` would run -- the SAME resolver.
        # With no saved dict at all, rung 4 (seed + this body chain) is
        # exactly the historical opening state, so only a calibration or a
        # model stamp changes anything. A saved dict that EXISTS but does not
        # apply (another variant, or no tile_fraction) is stale: its body /
        # fraction must not leak into this variant, so every origin applies
        # (M1: otherwise the dialog and the CLI open differently).
        opening_variant = self._variant.currentText()
        if not (
            "tile_fraction" in saved
            and str(saved.get("variant") or "") in ("", opening_variant)
        ):
            from hydra_suite.detectkit.jobs.semantic_escalation import (
                default_semantic_tiling,
            )

            chain_px = float(reference_body_px or 0.0)
            opening = default_semantic_tiling(
                project, opening_variant, body_chain_px=chain_px
            )
            stale_saved = bool(saved)
            if stale_saved or opening["origin"] in ("calibration", "stamped"):
                # r1: an unknown body (full_frame) zeroes only the body; the
                # fraction field keeps the seed, as with no saved dict, so
                # typing a body tiles exactly like `--reference-body-px N`.
                fraction = (
                    float(SEMANTIC_TILE_FRACTION_SEED)
                    if opening["origin"] == "full_frame"
                    else float(opening["tile_fraction"] or 0.0)
                )
                body = float(opening["reference_body_px"])
                if opening["origin"] == "calibration":
                    origin_label, body_source = "the model's calibration", "profile"
                elif opening["origin"] == "stamped" and chain_px <= 0:
                    origin_label, body_source = "stamped on the model", "stamped"
                else:
                    origin_label = body_px_origin or "entered by you"
                    body_source = _body_source(body_px_source, body_px_origin)
                self._body_origin_label.setText(origin_label)
                self._body_origin_label.setToolTip(origin_label)

        fraction = min(max(fraction, 0.0), FRACTION_MAX)
        tiling.set_spec(
            TilingSpec(
                enabled=fraction > 0.0,
                geometry_mode="auto_object",
                object_tile_fractions=(fraction,) if fraction > 0.0 else (),
                reference_body_px=max(body, 0.0),
                overlap=min(
                    max(_saved_value(saved, "overlap", DEFAULT_OVERLAP, float), 0.0),
                    OVERLAP_MAX,
                ),
                fragment_policy="crowd",
                merge_policy="nms",
                merge_metric="polygon_iou",
            ),
            extras={
                "seam_margin_px": _saved_value(
                    saved, "seam_margin_px", int(DEFAULT_SEAM_MARGIN_PX), int
                ),
                "merge_iou": _saved_value(saved, "merge_iou", DEFAULT_MERGE_IOU, float),
            },
        )
        tiling.set_reference_body(max(body, 0.0), body_source)
        form.addWidget(tiling, 2, 0, 1, 4)

        origin = QLabel(f"Body-size source: {self._body_origin_label.text()}")
        origin.setWordWrap(True)
        origin.setToolTip(self._body_origin_label.toolTip())
        # Keep this full-width provenance message on its own row below the
        # tiling group. Sharing a row made two widgets paint on top of one
        # another.
        form.addWidget(origin, 4, 0, 1, 4)
        self._body_origin_display = origin
        top.addWidget(settings_group, 5)
        outer.addLayout(top, 1)

        self._exhaustive = QCheckBox(
            "Labelled frames are exhaustive (every animal is marked)"
        )
        self._exhaustive.setChecked(bool(saved.get("exhaustive", False)))
        self._exhaustive.setToolTip(
            "Calibration counts an unlabelled real animal as a false positive, "
            "which biases the recommended threshold upward."
        )
        outer.addWidget(self._exhaustive)

        self._btn_calibrate = QPushButton("Calibrate against labelled frames…")
        self._btn_calibrate.setEnabled(False)
        self._btn_calibrate.clicked.connect(self._run_calibration)
        self._refresh_calibration_enabled()
        self._selector.calibration_changed.connect(self._refresh_calibration_enabled)

        self._btn_view_calibration = QPushButton("View saved calibration…")
        self._btn_view_calibration.clicked.connect(self._view_saved_calibration)

        self._btn_preview = QPushButton("Test random image…")
        self._btn_preview.setToolTip(
            "Chooses one random image from the selected sources, processes the "
            "complete image with the current tiling and confidence settings, "
            "then shows a zoomable prediction overlay and measured run-time "
            "estimate. No labels are written."
        )
        self._btn_preview.clicked.connect(self._run_preview)

        actions = QHBoxLayout()
        actions.addWidget(self._btn_calibrate, 2)
        actions.addWidget(self._btn_view_calibration, 1)
        actions.addWidget(self._btn_preview, 1)
        outer.addLayout(actions)

        # C1: the 3.45 GB download is surfaced HERE, before any run starts.
        # The tools-panel button is enabled when only the checkpoint is
        # missing precisely so this dialog can be reached to offer it.
        self._checkpoint_note = QLabel("")
        self._checkpoint_note.setWordWrap(True)
        outer.addWidget(self._checkpoint_note)
        self._refresh_checkpoint_note()
        self._variant.currentTextChanged.connect(
            lambda _t: self._refresh_checkpoint_note()
        )
        self._variant.currentTextChanged.connect(self.prefill_from_sidecar)

        self._scale_note = QLabel("")
        self._scale_note.setWordWrap(True)
        outer.addWidget(self._scale_note)
        self._scale_cache: dict = {}
        self._selector.calibration_changed.connect(self._refresh_scale_warning)
        self._selector.escalation_changed.connect(self._refresh_scale_warning)
        self._refresh_scale_warning()

        self._status = QLabel("")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        self._refresh_saved_calibration_ui()

        self.add_content(container)
        self.setMinimumSize(720, 500)
        self.resize(820, 560)
        self.fit_to_content(QSize(720, 500))
        self._tiling.btn_slice_advanced.toggled.connect(
            lambda _expanded: self.fit_to_content(QSize(720, 500))
        )

    # -- accessors used by the handler -------------------------------------

    def selected_sources(self) -> list:
        """The sources to ESCALATE."""
        return self._selector.escalation_sources()

    def calibration_sources(self) -> list:
        """The polygon ground-truth sources to calibrate on."""
        return self._selector.calibration_sources()

    def _refresh_scale_warning(self) -> None:
        self._scale_note.setText(
            scale_warning_text(
                self._scale_cache,
                self.calibration_sources(),
                self.selected_sources(),
            )
        )

    def selected_variant(self) -> str:
        return self._variant.currentText()

    def prefill_from_sidecar(self, model_key: str) -> None:
        """Default the prompt/tile-fraction fields from a published model's sidecar.

        A prefill, never a lock: ``sidecar_for`` returns ``None`` for a stock
        variant or an unregistered key (no-op), and the widgets stay enabled
        and editable either way -- REFERENCE_BODY_SIZE precedent, a measured
        value is sacrosanct but a derived one is only a starting point.
        """
        meta = sidecar_for(model_key)
        if meta is None:
            return
        prompt = meta.get("prompt")
        if prompt:
            self._prompt.setText(str(prompt))
        # Multi-scale artifacts omit the bare key and stamp their median as
        # `prefill_object_tile_fraction`; the shared reader accepts both, so a
        # multi-scale model prefills its own median instead of nothing.
        fraction = stamped_object_tile_fraction(meta)
        if fraction is not None:
            try:
                self._tile_fraction.setValue(float(fraction))
            except (TypeError, ValueError):
                pass
        # I3: the sidecar's reference_body_px is what this model was TRAINED
        # at (the tile scale). The project's reference_body_px is an
        # independent value (its own labels' median). Train/serve tile scale
        # can silently diverge if they disagree -- warn, but never
        # hard-refuse, since a deliberate re-scale is legitimate.
        # The comparison itself now lives in ``core.inference.geometry_drift``
        # so the SAM3/YOLO dataset builders and the headless ``--sahi-profile``
        # path share one guard instead of this dialog owning the only copy.
        # The user-facing wording stays HERE: core returns a typed verdict, and
        # each caller renders its own message from it.
        verdict = compare_geometry_value(
            "reference_body_px",
            meta.get("reference_body_px"),
            float(self._reference_body.value()),
        )
        # ``reference_body_px`` is a scalar from every publisher we own. The
        # shared guard also understands [w, h] tile pairs, which a spin box
        # cannot represent -- ignoring a non-scalar stamp here reproduces the
        # pre-extraction behaviour, where ``float(a_list)`` raised and the
        # whole block was skipped.
        if not isinstance(verdict.stamped_value, float):
            pass
        elif verdict.should_prefill:
            self._tiling.set_reference_body(verdict.stamped_value, "stamped")
        elif verdict.is_mismatch:
            QMessageBox.warning(
                self,
                "Body Size Mismatch",
                f"This model was trained with reference_body_px="
                f"{verdict.stamped_value:g}, but this project's Body size (px) "
                f"is {verdict.effective_value:g}. Train/serve tile scale can "
                "diverge silently if these disagree -- verify this is "
                "intentional (e.g. a deliberate re-scale) before running.",
            )
        self._refresh_tile_label()

    def prompt(self) -> str:
        return self._prompt.text().strip()

    def class_name(self) -> str:
        """The project class staged instances are labelled as.

        Empty when the project declares no classes; the job then falls back
        to the prompt, which is what pre-fix staging directories contain.
        """
        return self._class_name.currentText().strip()

    def reference_body_px(self) -> float:
        return float(self._reference_body.value())

    def parameters(self) -> dict:
        return {
            "class_name": self.class_name(),
            "device": self._device.device(),
            "confidence": float(self._confidence.value()),
            "max_instances": int(self._max_instances.value()),
            "overlap": float(self._overlap.value()),
            "seam_margin_px": float(self._seam_margin.value()),
            "merge_iou": float(self._merge_iou.value()),
            "reference_body_px": self.reference_body_px(),
            "tile_fraction": self.tile_fraction(),
            # The label-derived size gate from calibration. Not a control:
            # it is FITTED to the user's labels, so there is nothing to
            # type. 0/0 until a frontier point is chosen, which is exactly
            # the ungated behaviour of a run that skipped calibration.
            "area_min_px2": float(self._area_band[0]),
            "area_max_px2": float(self._area_band[1]),
        }

    @staticmethod
    def _restore_calibration_points(record: dict) -> list:
        from hydra_suite.core.inference.semantic.calibration import CalibrationPoint

        points = []
        for raw in record.get("points", []) if isinstance(record, dict) else []:
            try:
                points.append(CalibrationPoint(**dict(raw)))
            except (TypeError, ValueError):
                continue
        return points

    def _settings_payload(self) -> dict:
        return {
            "variant": self.selected_variant(),
            "prompt": self.prompt(),
            # Legacy key, still written so an older build reopens the dialog
            # with the same escalation selection.
            "source_names": [src.name for src in self.selected_sources()],
            **self._selector.state(),
            "exhaustive": self._exhaustive.isChecked(),
            # Persist 0.0 rather than None so QDoubleSpinBox can restore the
            # explicit full-frame choice without special-case coercion.
            "tile_fraction": float(self._tile_fraction.value()),
            **{
                key: value
                for key, value in self.parameters().items()
                if key != "tile_fraction"
            },
        }

    def _persist_settings(self) -> None:
        if self._project is None:
            return
        self._project.semantic_escalation_settings = self._settings_payload()
        if self._persist_callback is not None:
            self._persist_callback()

    def _store_calibration(
        self, points, recommended, reason: str, preview_frames=None
    ) -> None:
        self.calibration_points = list(points)
        self.calibration_preview_frames = (
            preview_frames if preview_frames is not None else []
        )
        # The band belongs to the LABELS, not to the chosen operating point:
        # every point in one sweep shares it. Adopting it here (rather than
        # only in apply_calibration_choice) stops a recalibration that the
        # user closes without picking a point from leaving the PREVIOUS
        # run's band in place, gating new data by old animal sizes.
        for point in points:
            bounds = (
                float(getattr(point, "area_min_px2", 0.0) or 0.0),
                float(getattr(point, "area_max_px2", 0.0) or 0.0),
            )
            if bounds[1] > bounds[0] > 0.0:
                self._area_band = bounds
            break
        if self._project is None:
            self._refresh_saved_calibration_ui()
            return
        recommended_index = -1
        if recommended is not None:
            recommended_index = next(
                (i for i, point in enumerate(points) if point is recommended), -1
            )
        saved = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "variant": self.selected_variant(),
            "prompt": self.prompt(),
            "source_names": [src.name for src in self.selected_sources()],
            "calibration_source_paths": self._selector.state()[
                "calibration_source_paths"
            ],
            "parameters": self.parameters(),
            "reason": str(reason or ""),
            "recommended_index": recommended_index,
            "points": [asdict(point) for point in points],
        }
        if self.calibration_preview_frames:
            from hydra_suite.detectkit.gui.calibration_preview_store import (
                save_calibration_previews,
            )

            saved["preview_artifact"] = save_calibration_previews(
                Path(self._project.project_dir), self.calibration_preview_frames
            )
        self._saved_calibration = saved
        # Per-PROJECT record, unchanged: this is what the dialog restores and
        # what an existing project already relies on.
        self._project.semantic_calibration = dict(self._saved_calibration)
        # Per-MODEL record (D10). Additive read-modify-write of the published
        # artifact's sidecar, so a headless SAM3 run sees the same operating
        # point. A stock variant has no sidecar and this is a logged no-op.
        # Note the scope difference: two projects calibrating one model are
        # last-write-wins HERE, while each keeps its own project copy.
        if write_serving_calibration(sidecar_path_for(self.selected_variant()), saved):
            self._calibration_origin = CalibrationOrigin.SIDECAR
        self._persist_settings()
        self._refresh_saved_calibration_ui()

    def _saved_recommendation(self):
        index = _saved_value(self._saved_calibration, "recommended_index", -1, int)
        if 0 <= index < len(self.calibration_points):
            return self.calibration_points[index]
        return None

    def _refresh_saved_calibration_ui(self) -> None:
        available = bool(self.calibration_points)
        self._btn_view_calibration.setEnabled(available)
        self._btn_calibrate.setText(
            "Recalibrate labelled frames…"
            if available
            else "Calibrate against labelled frames…"
        )
        if available and not self._status.text():
            created = str(self._saved_calibration.get("created_at", ""))[:10]
            when = f" from {created}" if created else ""
            # State WHERE it lives: a legacy project-only calibration is
            # invisible to a headless serving run, and the user cannot know
            # that unless it is said.
            where = getattr(self, "_calibration_origin", None)
            origin = f" [{where.label}]" if where is not None else ""
            self.set_status(
                f"Saved calibration{when}: {len(self.calibration_points)} "
                f"measured operating point(s).{origin}"
            )

    def _show_calibration_results(
        self,
        points,
        recommended,
        reason: str,
        *,
        partial: bool,
        preview_frames=None,
        merge_iou: float | None = None,
    ) -> None:
        from hydra_suite.detectkit.gui.dialogs.calibration_results_dialog import (
            CalibrationResultsDialog,
        )

        results = CalibrationResultsDialog(
            points,
            recommended,
            reason,
            project_frames=self._project_frame_count(),
            partial=partial,
            preview_frames=preview_frames,
            merge_iou=float(
                self.parameters()["merge_iou"] if merge_iou is None else merge_iou
            ),
            parent=self,
        )
        results.exec()
        chosen = results.chosen()
        if chosen is None:
            self.set_status(reason or "Calibration finished; no point chosen.")
            return
        self.apply_calibration_choice(chosen)
        self._persist_settings()
        tile_desc = (
            "full frame"
            if chosen.tile_fraction is None
            else f"tile fraction {chosen.tile_fraction:.2f} "
            f"({chosen.tile_px} px, {chosen.tiles_per_frame} tiles/frame)"
        )
        self.set_status(
            f"Using {tile_desc} at confidence {chosen.confidence:.2f}: misses "
            f"{chosen.missed_per_frame:.1f} animal(s)/frame, leaves "
            f"{chosen.extra_per_frame:.1f} polygon(s)/frame to delete "
            f"(recall {chosen.recall:.1%}, {chosen.n_matched} matched, "
            f"{chosen.seconds_per_frame:.1f} s/frame measured here)."
        )

    def _view_saved_calibration(self) -> None:
        if not self.calibration_points:
            return
        if not self.calibration_preview_frames and self._project is not None:
            artifact = str(self._saved_calibration.get("preview_artifact", ""))
            if artifact:
                from hydra_suite.detectkit.gui.calibration_preview_store import (
                    load_calibration_previews,
                )

                self.calibration_preview_frames = load_calibration_previews(
                    Path(self._project.project_dir), artifact
                )
        self._show_calibration_results(
            self.calibration_points,
            self._saved_recommendation(),
            str(self._saved_calibration.get("reason", "")),
            partial=False,
            preview_frames=self.calibration_preview_frames,
            merge_iou=_saved_value(
                dict(self._saved_calibration.get("parameters", {}) or {}),
                "merge_iou",
                self.parameters()["merge_iou"],
                float,
            ),
        )

    def _refresh_tile_label(self) -> None:
        self._tiling.refresh()

    def _project_frame_count(self) -> int:
        """Images across the selected sources — the run-time projection base."""
        return self._frame_count_for(self.selected_sources() or self._sources)

    @staticmethod
    def _frame_count_for(sources) -> int:
        from hydra_suite.detectkit.gui.constants import IMG_EXTS

        total = 0
        for src in sources:
            images = Path(src.path) / "images"
            if images.is_dir():
                total += sum(
                    1
                    for p in images.rglob("*")
                    if p.suffix.lower() in IMG_EXTS and not is_hidden_file(p)
                )
        return total

    def tile_fraction(self) -> float | None:
        value = float(self._tile_fraction.value())
        return None if value <= 0.0 else value

    def apply_calibration_choice(self, point) -> None:
        """Write a chosen frontier point back into the dialog's controls."""
        self._area_band = (
            float(getattr(point, "area_min_px2", 0.0) or 0.0),
            float(getattr(point, "area_max_px2", 0.0) or 0.0),
        )
        self._confidence.setValue(float(point.confidence))
        self._tile_fraction.setValue(
            0.0 if point.tile_fraction is None else float(point.tile_fraction)
        )
        self._refresh_tile_label()

    def set_calibration_enabled(self, enabled: bool, reason: str = "") -> None:
        self._btn_calibrate.setEnabled(enabled)
        if not enabled and reason:
            self._btn_calibrate.setToolTip(reason)

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def _refresh_calibration_enabled(self) -> None:
        # The selector found the polygon sources with has_polygon_frames, a
        # label-FILE scan: no image is decoded on the GUI thread to answer it.
        if not self._selector.has_calibration_sources():
            self.set_calibration_enabled(False, NO_POLYGON_SOURCES)
            return
        self.set_calibration_enabled(
            bool(self.calibration_sources()),
            'Select at least one source under "Calibrate on".',
        )

    def _run_calibration(self) -> None:
        from PySide6.QtWidgets import QProgressDialog

        from hydra_suite.core.inference.semantic.calibration import recommend
        from hydra_suite.detectkit.jobs.semantic_escalation import CalibrationWorker

        if not self._exhaustive.isChecked():
            QMessageBox.information(
                self,
                "Calibrate",
                "Confirm your labelled frames are exhaustively labelled first. "
                "An unlabelled real animal counts as a false positive and biases "
                "the recommended threshold upward.",
            )
            return
        sources = self.calibration_sources()
        if not sources:
            QMessageBox.information(
                self, "Calibrate", 'Select a source under "Calibrate on".'
            )
            return
        if self.calibration_points:
            created = str(self._saved_calibration.get("created_at", ""))[:10]
            suffix = f" from {created}" if created else ""
            reply = QMessageBox.question(
                self,
                "Replace saved calibration?",
                "A saved calibration"
                f"{suffix} already exists. Completing a new calibration will "
                "replace its measured frontier and recommendation. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        if not self.confirm_checkpoint():
            return

        self._persist_settings()

        # F4: the progress dialog exists BEFORE any decoding starts. Reading
        # the labelled frames is itself a cv2.imread of every labelled image
        # of every selected source, so it belongs behind this, in the worker,
        # under Cancel -- not on the GUI thread with the window frozen.
        progress = QProgressDialog("Reading labelled frames…", "Cancel", 0, 100, self)
        progress.setWindowTitle("SAM3 calibration")
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setModal(True)
        progress.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        progress.setMinimumWidth(420)
        worker = CalibrationWorker(
            sources,
            self.prompt(),
            self.selected_variant(),
            self.parameters(),
            project_dir=(
                self._project.project_dir if self._project is not None else None
            ),
        )
        progress.canceled.connect(worker.cancel)
        worker.progress.connect(progress.setValue)
        worker.status.connect(progress.setLabelText)

        def _done(points) -> None:
            # Read BEFORE close(): QProgressDialog.close() emits `canceled`,
            # which is wired to worker.cancel, so every completed sweep used
            # to look cancelled here and was never stored.
            cancelled = worker.cancelled
            progress.close()
            if not points:
                self.set_status(
                    "Calibration produced nothing: no polygon ground-truth "
                    "frames were found in the calibration source(s), or it was "
                    "cancelled before the first frame finished."
                )
                return
            best, reason = recommend(points)
            # A cancelled/partial sweep is useful to inspect, but must not erase
            # the last complete calibration stored with the project.
            if not cancelled:
                self._store_calibration(
                    points, best, reason, preview_frames=worker.preview_frames
                )
            self._show_calibration_results(
                points,
                best,
                reason,
                partial=cancelled,
                preview_frames=worker.preview_frames,
            )

        worker.result_ready.connect(_done)
        worker.finished.connect(progress.close)
        self._calibration_worker = worker  # keep a reference alive
        progress.show()
        progress.raise_()
        progress.activateWindow()
        worker.start()

    # -- checkpoint download, surfaced before anything runs -----------------

    def _refresh_checkpoint_note(self) -> None:
        # probe_checkpoint, NOT probe_availability: the model dropdown now
        # offers published finetuned registry keys alongside stock variants,
        # and probe_availability rejects anything outside SAM3_VARIANTS as
        # "Unknown SAM3 variant" -- which made every finetuned model
        # unselectable. probe_checkpoint handles both key spaces.
        avail = probe_checkpoint(self.selected_variant())
        if avail.checkpoint_missing:
            self._checkpoint_note.setText(
                f"⚠ The {self.selected_variant()} checkpoint "
                f"(~{CHECKPOINT_SIZE_GB:.2f} GB) is not on this machine yet. It "
                "will be downloaded once, after you confirm, before the run "
                "starts."
            )
        elif not avail.usable:
            self._checkpoint_note.setText(f"⚠ {avail.reason}")
        else:
            self._checkpoint_note.setText("")

    def confirm_checkpoint(self) -> bool:
        """Ask before a 3.45 GB download. True = go ahead."""
        avail = probe_checkpoint(self.selected_variant())
        if avail.usable:
            return True
        if not avail.checkpoint_missing:
            QMessageBox.warning(self, "Semantic escalation", avail.reason)
            return False
        reply = QMessageBox.question(
            self,
            "Download the SAM3 checkpoint?",
            f"The {self.selected_variant()} checkpoint (~{CHECKPOINT_SIZE_GB:.2f} "
            "GB) has not been downloaded yet.\n\nIt will be downloaded once "
            "and cached; the run cannot start without it. Download now?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    # -- random complete-frame preview -------------------------------------

    def _run_preview(self) -> None:
        from PySide6.QtWidgets import QProgressDialog

        from hydra_suite.detectkit.jobs.semantic_escalation import FramePreviewWorker

        if not self.prompt():
            QMessageBox.information(self, "Test random image", "Enter a prompt first.")
            return
        sources = self.selected_sources()
        if not sources:
            QMessageBox.information(self, "Test random image", "Select a source.")
            return
        if not self.confirm_checkpoint():
            return

        self._persist_settings()

        selected_frames = self._frame_count_for(sources)
        project_frames = self._frame_count_for(self._sources)
        progress = QProgressDialog("Choosing a random image…", "Cancel", 0, 100, self)
        progress.setWindowTitle("SAM3 complete-frame check")
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setModal(True)
        self._btn_preview.setEnabled(False)

        def _done(res) -> None:
            progress.close()
            self.set_status(
                f"Tested complete image {res.image_path.name}: "
                f"{len(res.predictions)} prediction(s) in {res.seconds:.1f} s."
            )
            from hydra_suite.detectkit.gui.dialogs.semantic_frame_preview_dialog import (
                SemanticFramePreviewDialog,
            )

            preview = SemanticFramePreviewDialog(
                res,
                selected_frames=selected_frames,
                project_frames=project_frames,
                parent=self,
            )
            preview.exec()

        def _failed(msg: str) -> None:
            cancelled = worker.cancelled  # before close(): it emits `canceled`
            progress.close()
            if cancelled:
                self.set_status("Random image check cancelled.")
                return
            QMessageBox.warning(self, "Test random image", msg)

        worker = FramePreviewWorker(
            sources, self.prompt(), self.selected_variant(), self.parameters()
        )
        progress.canceled.connect(worker.cancel)
        worker.progress.connect(progress.setValue)
        worker.status.connect(progress.setLabelText)
        worker.result_ready.connect(_done)
        worker.error.connect(_failed)
        worker.finished.connect(lambda: self._btn_preview.setEnabled(True))
        worker.finished.connect(progress.close)
        self._preview_worker = worker  # keep a reference alive
        progress.show()
        progress.raise_()
        progress.activateWindow()
        worker.start()

    def accept(self) -> None:  # noqa: D102
        if not self.prompt():
            QMessageBox.warning(self, "Semantic escalation", "Enter a prompt first.")
            return
        if not self.selected_sources():
            QMessageBox.warning(self, "Semantic escalation", "Select a source.")
            return
        if not self.confirm_checkpoint():
            return
        self._persist_settings()
        super().accept()
