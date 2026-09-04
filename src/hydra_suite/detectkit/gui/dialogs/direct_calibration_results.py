"""The direct-calibration frontier: every measured row, with overlays.

This is the only surface where a user turns measured SAHI evidence into a
NAMED, saved profile on the model artifact. Nothing is written until
``accept()`` -- profiles are staged in memory (``self._staged_meta``) and
committed with one atomic ``write_slice_meta`` call, so closing the dialog
without confirmation saves nothing. Scoring, recommendation and profile
mutation all live in core; this dialog only calls them and renders their
output.

A calibration done here AFTER a model is already registered edits the
sidecar and then refreshes that artifact's registry ``slice_profiles``
summary in the same ``accept()``, so the two can never disagree (see
``training.model_publish.verify_profile_summary``, which is asserted right
after the refresh). The sidecar remains the source of truth on read; the
registry summary is an inventory, never a second authority.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.core.inference.direct_calibration import (
    RECOMMENDATION_RULE,
    recommend_balanced,
)
from hydra_suite.core.inference.direct_calibration_grid import checkpoint_fingerprint
from hydra_suite.core.inference.slice_meta import (
    profile_summary,
    read_slice_meta,
    remove_slice_profile,
    upsert_slice_profile,
    write_slice_meta,
)
from hydra_suite.detectkit.gui.canvas import OBBCanvas
from hydra_suite.widgets.dialogs import BaseDialog

from ._overlay_helpers import dialog_gt_layer, dialog_pred_layer

_LOGGER = logging.getLogger(__name__)

# Column layout for the measured-row table. Declared as module constants so
# tests can assert on them without hard-coding integers.
COL_LABEL = 0
COL_FULL_FRAME = 1
COL_TILE_SIZE = 2
COL_OVERLAP = 3
COL_CONFIDENCE = 4
COL_MERGE = 5
COL_MAX_DETECTIONS = 6
COL_TILES_PER_FRAME = 7
COL_SECONDS_PER_FRAME = 8
COL_PROJECTED_DURATION = 9
COL_MATCHED = 10
COL_MISSED = 11
COL_EXTRA = 12
COL_DUPLICATE = 13
COL_PRECISION = 14
COL_RECALL = 15
COL_F1 = 16
COL_LOCALIZATION_QUALITY = 17
COL_FRAMES_INSTANCES = 18
COL_STATUS = 19

_COLUMN_LABELS = [
    "Candidate",
    "Full frame?",
    "Tile size",
    "Overlap",
    "Confidence",
    "Merge",
    "Max det.",
    "Tiles/frame",
    "s/frame",
    "Projected duration",
    "Matched",
    "Missed",
    "Extra",
    "Duplicate",
    "Precision",
    "Recall",
    "F1",
    "Localization quality",
    "Frames / instances",
    "Status",
]


def _close(left: float, right: float) -> bool:
    """Float-safe equality for a merge threshold read back from JSON."""
    return abs(left - right) <= 1e-9


def _humanise_duration(seconds: float, frames: int) -> str:
    total = seconds * max(frames, 0)
    total_int = int(round(total))
    hours, rem = divmod(total_int, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours} h {minutes:02d} m"
    if minutes:
        return f"{minutes} m {secs:02d} s"
    return f"{secs} s"


class DirectCalibrationResultsDialog(BaseDialog):
    """Show the full measured frontier and stage named profiles for saving."""

    COL_LABEL = COL_LABEL
    COL_FULL_FRAME = COL_FULL_FRAME
    COL_TILE_SIZE = COL_TILE_SIZE
    COL_OVERLAP = COL_OVERLAP
    COL_CONFIDENCE = COL_CONFIDENCE
    COL_MERGE = COL_MERGE
    COL_MAX_DETECTIONS = COL_MAX_DETECTIONS
    COL_TILES_PER_FRAME = COL_TILES_PER_FRAME
    COL_SECONDS_PER_FRAME = COL_SECONDS_PER_FRAME
    COL_PROJECTED_DURATION = COL_PROJECTED_DURATION
    COL_MATCHED = COL_MATCHED
    COL_MISSED = COL_MISSED
    COL_EXTRA = COL_EXTRA
    COL_DUPLICATE = COL_DUPLICATE
    COL_PRECISION = COL_PRECISION
    COL_RECALL = COL_RECALL
    COL_F1 = COL_F1
    COL_LOCALIZATION_QUALITY = COL_LOCALIZATION_QUALITY
    COL_FRAMES_INSTANCES = COL_FRAMES_INSTANCES
    COL_STATUS = COL_STATUS

    def __init__(
        self,
        parent,
        *,
        model_path,
        outcome,
        training_geometry: dict[str, Any],
        previews: list | None = None,
        task: str = "obb",
        class_names=None,
        runtime_tier: str = "",
        evidence_split: str = "",
        label_set_fingerprint: str = "",
    ) -> None:
        super().__init__(
            "SAHI calibration results",
            parent,
            buttons=QDialogButtonBox.Save | QDialogButtonBox.Cancel,
        )
        self._model_path = Path(model_path)
        self.outcome = outcome
        self._training_geometry = dict(training_geometry)
        self._previews = list(previews or [])
        self._task = task
        self._class_names = class_names
        # Measurement provenance carried from the request/evidence, so a
        # profile records the tier it was timed on and the labels it was
        # scored against -- not the literal string "measured".
        self._runtime_tier = str(runtime_tier or "")
        self._evidence_split = str(evidence_split or "")
        self._label_set_fingerprint = str(label_set_fingerprint or "")
        self._frame_index = 0

        self._recommended_point, self._recommendation_reason = recommend_balanced(
            outcome.points
        )

        existing = read_slice_meta(self._model_path)
        self._staged_meta: dict[str, Any] = (
            existing
            if existing is not None
            else {"training_geometry": self._training_geometry}
        )
        # Snapshot for accept()'s "nothing was staged" guard -- comparing by
        # value, not identity, since save/remove/primary-change all rebind
        # self._staged_meta to a new dict.
        self._initial_meta_snapshot = dict(self._staged_meta)
        self._updating_primary_combo = False

        self._build_ui()
        self._populate_table()
        self._refresh_profile_list()
        if outcome.points:
            self.table_rows.selectRow(0)
        self._render_preview()
        self._update_save_enabled()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        container = QWidget()
        outer = QVBoxLayout(container)

        rule_label = QLabel(f"Recommendation rule: {RECOMMENDATION_RULE}")
        rule_label.setWordWrap(True)
        outer.addWidget(rule_label)

        reason_label = QLabel(self._recommendation_reason)
        reason_label.setWordWrap(True)
        outer.addWidget(reason_label)

        splitter = QSplitter(Qt.Orientation.Vertical)

        self.table_rows = QTableWidget(0, len(_COLUMN_LABELS))
        self.table_rows.setHorizontalHeaderLabels(_COLUMN_LABELS)
        self.table_rows.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_rows.setSelectionMode(QTableWidget.SingleSelection)
        self.table_rows.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_rows.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.table_rows.currentCellChanged.connect(self._on_row_changed)
        splitter.addWidget(self.table_rows)

        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 8, 0, 0)

        preview_header = QHBoxLayout()
        self.btn_prev_frame = QPushButton("< Previous frame")
        self.btn_prev_frame.clicked.connect(lambda: self._step_frame(-1))
        preview_header.addWidget(self.btn_prev_frame)
        self._frame_label = QLabel("")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_header.addWidget(self._frame_label, 1)
        self.btn_next_frame = QPushButton("Next frame >")
        self.btn_next_frame.clicked.connect(lambda: self._step_frame(1))
        preview_header.addWidget(self.btn_next_frame)
        self.chk_show_gt = QCheckBox("Ground truth")
        self.chk_show_gt.setChecked(True)
        self.chk_show_gt.toggled.connect(self._refresh_visibility)
        preview_header.addWidget(self.chk_show_gt)
        self.chk_show_pred = QCheckBox("Predictions")
        self.chk_show_pred.setChecked(True)
        self.chk_show_pred.toggled.connect(self._refresh_visibility)
        preview_header.addWidget(self.chk_show_pred)
        preview_layout.addLayout(preview_header)

        self.canvas = OBBCanvas()
        self.canvas.setMinimumHeight(260)
        preview_layout.addWidget(self.canvas, 1)
        # The overlay must state which operating point it depicts -- a user
        # saves a profile from what they SEE here.
        self.lbl_overlay_caption = QLabel("")
        self.lbl_overlay_caption.setWordWrap(True)
        preview_layout.addWidget(self.lbl_overlay_caption)
        splitter.addWidget(preview)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        outer.addWidget(splitter, 1)

        entry_row = QHBoxLayout()
        self.edit_profile_name = QLineEdit()
        self.edit_profile_name.setPlaceholderText("Balanced")
        self.edit_profile_name.textChanged.connect(self._update_save_enabled)
        entry_row.addWidget(QLabel("Name:"))
        entry_row.addWidget(self.edit_profile_name, 1)
        self.edit_profile_note = QLineEdit()
        self.edit_profile_note.setPlaceholderText("Purpose note (optional)")
        entry_row.addWidget(QLabel("Note:"))
        entry_row.addWidget(self.edit_profile_note, 1)
        self.chk_make_primary = QCheckBox("Make primary")
        self.chk_make_primary.setChecked(False)
        entry_row.addWidget(self.chk_make_primary)
        self.btn_save_profile = QPushButton("Save as profile")
        self.btn_save_profile.clicked.connect(self._on_save_profile_clicked)
        entry_row.addWidget(self.btn_save_profile)
        outer.addLayout(entry_row)

        save_row = QHBoxLayout()
        # The combo SELECTS a staged profile; it does not designate one.
        # Wiring designation to currentIndexChanged meant that merely picking
        # a profile in order to remove it silently made that profile primary
        # (and then the removal always hit the "you removed the primary"
        # prompt). Primary is only ever an explicit decision -- the button
        # below is that decision.
        self.combo_primary = QComboBox()
        save_row.addWidget(QLabel("Staged profiles (primary marked):"))
        save_row.addWidget(self.combo_primary, 1)
        self.btn_set_primary = QPushButton("Set as primary")
        self.btn_set_primary.clicked.connect(self._on_set_primary_clicked)
        save_row.addWidget(self.btn_set_primary)
        self.btn_remove_profile = QPushButton("Remove selected")
        self.btn_remove_profile.clicked.connect(self._on_remove_profile_clicked)
        save_row.addWidget(self.btn_remove_profile)
        outer.addLayout(save_row)

        self.add_content(container)

    # ------------------------------------------------------------------
    # Table population
    # ------------------------------------------------------------------

    def _populate_table(self) -> None:
        points = self.outcome.points
        self.table_rows.setRowCount(len(points))
        for row, point in enumerate(points):
            status = ""
            if point.failed_reason:
                status = point.failed_reason
            elif point is self._recommended_point:
                status = "Recommended"
            values = {
                COL_LABEL: point.label,
                COL_FULL_FRAME: "Yes" if not point.enabled else "No",
                COL_TILE_SIZE: f"{point.tile_width}x{point.tile_height}",
                COL_OVERLAP: f"{point.overlap:g}",
                COL_CONFIDENCE: f"{point.confidence:g}",
                COL_MERGE: (
                    f"{point.merge_policy}/{point.merge_metric}"
                    f"@{point.merge_threshold:g} ({point.merge_backend})"
                ),
                COL_MAX_DETECTIONS: str(point.max_detections),
                COL_TILES_PER_FRAME: str(point.tiles_per_frame),
                COL_SECONDS_PER_FRAME: f"{point.seconds_per_frame:.3f}",
                COL_PROJECTED_DURATION: _humanise_duration(
                    point.seconds_per_frame, point.score.frames
                ),
                COL_MATCHED: str(point.score.matched),
                COL_MISSED: str(point.score.missed),
                COL_EXTRA: str(point.score.extra),
                COL_DUPLICATE: str(point.score.duplicate),
                COL_PRECISION: f"{point.score.precision:.3f}",
                COL_RECALL: f"{point.score.recall:.3f}",
                COL_F1: f"{point.score.f1:.3f}",
                COL_LOCALIZATION_QUALITY: f"{point.score.mean_iou:.3f}",
                COL_FRAMES_INSTANCES: (
                    f"{point.score.frames}f / "
                    f"{point.score.matched + point.score.missed}i"
                ),
                COL_STATUS: status,
            }
            for col, text in values.items():
                self.table_rows.setItem(row, col, QTableWidgetItem(text))

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _current_point(self):
        row = self.table_rows.currentRow()
        if row < 0 or row >= len(self.outcome.points):
            return None
        return self.outcome.points[row]

    def _preview_for_point(self, point):
        """The preview measured at THIS row's geometry, merge AND confidence.

        Keyed by ``(candidate_index, merge_threshold, confidence)`` and never
        by label: candidate labels are not unique (the grid dedups on
        geometry), and a label lookup is exactly what made every row of a
        geometry show the same permissive overlay. Confidence is part of the
        key because it is NOT reproducible from a permissive preview -- it
        gates detections before the raw cap and the merge, both of which
        change which polygons exist.
        """
        if point is None:
            return None
        for preview in self._previews:
            if int(getattr(preview, "candidate_index", -1)) != int(
                point.candidate_index
            ):
                continue
            if not _close(
                float(getattr(preview, "merge_threshold", -1.0)),
                float(point.merge_threshold),
            ):
                continue
            if not _close(
                float(getattr(preview, "confidence", -1.0)),
                float(point.confidence),
            ):
                continue
            return preview
        return None

    @staticmethod
    def _row_predictions(preview, point, frame_index):
        """The selected row's post-merge predictions, verbatim.

        The preview was collected by ``_preview_for`` at this row's own
        geometry, merge threshold, confidence AND ``max_targets``, so it IS
        the row's output. Nothing is replayed here: the previous replay
        (confidence gate + largest-first cap over a permissive preview) could
        not be exact, because ``max_targets`` also derives a raw cap applied
        by confidence around the merge.
        """
        _path, _gt, pred_polygons = preview.frames[frame_index]
        return list(pred_polygons)

    def _overlay_caption(self, point) -> str:
        if point is None:
            return ""
        return (
            f"Overlay depicts: {point.label} · confidence >= "
            f"{point.confidence:g} · merge {point.merge_policy}/"
            f"{point.merge_metric}@{point.merge_threshold:g} · max "
            f"{point.max_detections} detections/frame."
        )

    def _on_row_changed(self, *_unused) -> None:
        self._frame_index = 0
        self._render_preview()
        self._update_save_enabled()

    def _step_frame(self, amount: int) -> None:
        preview = self._preview_for_point(self._current_point())
        if preview is None or not preview.frames:
            return
        self._frame_index = (self._frame_index + amount) % len(preview.frames)
        self._render_preview()

    def _refresh_visibility(self) -> None:
        self.canvas.set_layer_visible("gt", self.chk_show_gt.isChecked())
        self.canvas.set_layer_visible("pred", self.chk_show_pred.isChecked())

    def _render_preview(self) -> None:
        """Render stored polygons only. Never runs inference/the model."""
        point = self._current_point()
        preview = self._preview_for_point(point)
        has_frames = bool(preview and preview.frames)
        self.btn_prev_frame.setEnabled(has_frames and len(preview.frames) > 1)
        self.btn_next_frame.setEnabled(has_frames and len(preview.frames) > 1)
        self.chk_show_gt.setEnabled(has_frames)
        self.chk_show_pred.setEnabled(has_frames)
        self.lbl_overlay_caption.setText(self._overlay_caption(point))
        if not has_frames:
            self.canvas.clear_all()
            self._frame_label.setText("No visual preview")
            return
        self._frame_index %= len(preview.frames)
        image_path, gt_polygons, _all_pred = preview.frames[self._frame_index]
        pred_polygons = self._row_predictions(preview, point, self._frame_index)
        self._frame_label.setText(
            f"Frame {self._frame_index + 1} of {len(preview.frames)} · "
            f"{Path(image_path).name}"
        )
        self.canvas.load_image(str(image_path))
        class_names = self._class_names or ["object"]
        gt_detections = [
            {"class_id": 0, "polygon_px": polygon, "confidence": 1.0}
            for polygon in gt_polygons
        ]
        pred_detections = [
            {"class_id": 0, "polygon_px": polygon, "confidence": 1.0}
            for polygon in pred_polygons
        ]
        self.canvas.set_layer(dialog_gt_layer(gt_detections, class_names))
        self.canvas.set_layer(dialog_pred_layer(pred_detections, class_names))
        self._refresh_visibility()

    # ------------------------------------------------------------------
    # Settings / measurement payloads
    # ------------------------------------------------------------------

    def settings_for_row(self, row: int) -> dict[str, Any]:
        point = self.outcome.points[row]
        return {
            "enabled": bool(point.enabled),
            "geometry_mode": point.geometry_mode,
            "slice_width": int(point.tile_width),
            "slice_height": int(point.tile_height),
            "overlap": float(point.overlap),
            "object_tile_fraction": float(point.object_tile_fraction),
            "trained_body_px": float(
                self._training_geometry.get("reference_body_px", 0.0) or 0.0
            ),
            "confidence_threshold": float(point.confidence),
            "merge_policy": point.merge_policy,
            "merge_metric": point.merge_metric,
            "merge_threshold": float(point.merge_threshold),
            "merge_backend": point.merge_backend,
            "max_detections": int(point.max_detections),
        }

    def measurement_for_row(self, row: int) -> dict[str, Any]:
        point = self.outcome.points[row]
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_fingerprint": checkpoint_fingerprint(self._model_path),
            "task": self._task,
            "frames": int(point.score.frames),
            "instances": int(point.score.matched + point.score.missed),
            "label_set_fingerprint": self._label_set_fingerprint,
            "split": self._evidence_split,
            # The runtime TIER the sweep was timed on (cpu/gpu/gpu_fast).
            # Timings are measurements on this data and this tier, never
            # portable guarantees.
            "runtime": self._runtime_tier,
            "merge_backend": str(point.merge_backend or ""),
            "seconds_per_frame": float(point.seconds_per_frame),
            "precision": float(point.score.precision),
            "recall": float(point.score.recall),
            "f1": float(point.score.f1),
            "localization_quality": float(point.score.mean_iou),
            "max_detections": int(point.max_detections),
        }

    # ------------------------------------------------------------------
    # Staged profile mutation (in memory only until accept())
    # ------------------------------------------------------------------

    def save_profile(self, name: str, note: str = "", primary: bool = False) -> None:
        row = self.table_rows.currentRow()
        if row < 0 or row >= len(self.outcome.points):
            raise ValueError("No row is selected to save as a profile.")
        point = self.outcome.points[row]
        if point.failed_reason:
            raise ValueError(
                f"Row {point.label!r} failed ({point.failed_reason}) and cannot "
                "become a profile."
            )
        self._staged_meta = upsert_slice_profile(
            self._staged_meta,
            name=name,
            settings=self.settings_for_row(row),
            note=note,
            measurement=self.measurement_for_row(row),
            primary=primary,
        )
        self._refresh_profile_list()

    def remove_profile(
        self, profile_id: str, new_primary_id: str | None = None
    ) -> None:
        self._staged_meta = remove_slice_profile(
            self._staged_meta, profile_id, new_primary_id=new_primary_id
        )
        self._refresh_profile_list()

    def staged_profiles(self) -> list[dict[str, Any]]:
        return list(self._staged_meta.get("profiles") or [])

    def staged_meta(self) -> dict[str, Any]:
        return self._staged_meta

    def _refresh_profile_list(self) -> None:
        # Guard against signal recursion while repopulating -- the same
        # pattern the codebase's other combos use (see training_dialog.py's
        # blockSignals around programmatic rebuilds).
        self._updating_primary_combo = True
        try:
            self.combo_primary.clear()
            primary_id = self._staged_meta.get("primary_profile_id", "")
            selected_index = -1
            for index, profile in enumerate(self.staged_profiles()):
                label = profile["name"]
                if profile["id"] == primary_id:
                    label += " (primary)"
                    selected_index = index
                self.combo_primary.addItem(label, profile["id"])
            if selected_index >= 0:
                self.combo_primary.setCurrentIndex(selected_index)
        finally:
            self._updating_primary_combo = False
        self._update_save_enabled()

    # ------------------------------------------------------------------
    # Widget wiring (save / remove / primary selection)
    # ------------------------------------------------------------------

    def _update_save_enabled(self) -> None:
        if not hasattr(self, "btn_save_profile"):
            return
        point = self._current_point()
        name_present = bool(self.edit_profile_name.text().strip())
        row_ok = point is not None and not point.failed_reason
        self.btn_save_profile.setEnabled(name_present and row_ok)

    def _on_save_profile_clicked(self) -> None:
        name = self.edit_profile_name.text().strip()
        note = self.edit_profile_note.text()
        primary = self.chk_make_primary.isChecked()
        try:
            self.save_profile(name, note=note, primary=primary)
        except ValueError as exc:
            QMessageBox.warning(self, "Could not save profile", str(exc))
            return
        self.edit_profile_name.clear()
        self.edit_profile_note.clear()
        self.chk_make_primary.setChecked(False)

    def _prompt_replacement_primary(
        self, candidates: list[dict[str, Any]]
    ) -> str | None:
        """Ask the user which staged profile becomes primary, or to clear it.

        Returns the chosen profile id, ``""`` to clear the designation, or
        ``None`` if the user cancelled (in which case the removal must not
        proceed). Never picks a replacement on the user's behalf -- that is
        exactly what Task 6 removed from the core.
        """
        options = ["(clear primary designation)"] + [c["name"] for c in candidates]
        choice, accepted = QInputDialog.getItem(
            self,
            "Choose a new primary profile",
            "The profile you removed was the primary. Pick its replacement:",
            options,
            editable=False,
        )
        if not accepted:
            return None
        if choice == options[0]:
            return ""
        index = options.index(choice) - 1
        return candidates[index]["id"]

    def _on_remove_profile_clicked(self) -> None:
        index = self.combo_primary.currentIndex()
        if index < 0:
            return
        profile_id = self.combo_primary.itemData(index)
        if not profile_id:
            return
        try:
            self.remove_profile(profile_id)
        except ValueError:
            remaining = [p for p in self.staged_profiles() if p["id"] != profile_id]
            chosen = self._prompt_replacement_primary(remaining)
            if chosen is None:
                return
            self.remove_profile(profile_id, new_primary_id=chosen)

    def _on_set_primary_clicked(self) -> None:
        self.set_primary(self.combo_primary.currentData())

    def set_primary(self, profile_id: str | None) -> None:
        """Designate ``profile_id`` primary. Only ever called by an explicit act."""
        if self._updating_primary_combo or not profile_id:
            return
        if self._staged_meta.get("primary_profile_id") == profile_id:
            return
        # Re-designate primary without touching any profile's settings --
        # upsert_slice_profile's contract (primary is only ever an explicit
        # decision) is preserved by re-upserting the same profile unchanged
        # with primary=True.
        profile = next(p for p in self.staged_profiles() if p["id"] == profile_id)
        self._staged_meta = upsert_slice_profile(
            self._staged_meta,
            name=profile["name"],
            settings=profile["settings"],
            profile_id=profile["id"],
            note=profile["note"],
            measurement=profile["measurement"],
            primary=True,
        )
        self._refresh_profile_list()

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def _refresh_registry_profile_summary(self) -> None:
        """Keep the registry's ``slice_profiles`` summary equal to the sidecar.

        Post-publish calibration writes only the sidecar, so without this the
        registry summary is permanently stale -- and the spec's rule is that
        any disagreement between the two is a visible failure, never a second
        source of truth. If the calibrated checkpoint is not a registered
        artifact there is nothing to refresh and this does nothing.
        """
        try:
            from hydra_suite.training.model_publish import (
                _registry_key_for_model,
                load_model_registry,
                save_model_registry,
                verify_profile_summary,
            )

            registry = load_model_registry() or {}
            entries = registry.get("entries")
            if not isinstance(entries, dict):
                return
            key = _registry_key_for_model(self._model_path)
            entry = entries.get(key)
            if not isinstance(entry, dict):
                return
            entry["slice_profiles"] = profile_summary(self._staged_meta)
            save_model_registry(registry)
            verify_profile_summary(self._model_path, entry["slice_profiles"])
        except Exception:
            _LOGGER.exception(
                "Could not refresh the registry's SAHI profile summary for %s; "
                "the sidecar remains the source of truth.",
                self._model_path,
            )

    def accept(self) -> None:
        if self._staged_meta != self._initial_meta_snapshot:
            write_slice_meta(self._model_path, self._staged_meta)
            self._refresh_registry_profile_summary()
        super().accept()

    def reject(self) -> None:
        super().reject()
