"""The calibration gate: what will be measured, and what it will cost.

Before any model runs, the user must see the evidence that will be scored,
the exact candidate grid, and its tile cost, and must affirm their labels
are exhaustive -- an unlabelled real animal looks like a false positive and
biases calibration toward settings that are too strict. This wizard builds
that affirmation and assembles the ``DirectCalibrationRequest`` the sweep
consumes; it holds no scoring or grid logic of its own.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.core.inference.direct_calibration import RECOMMENDATION_RULE
from hydra_suite.core.inference.direct_calibration_grid import (
    DEFAULT_MAX_TOTAL_TILES,
    build_candidate_grid,
    estimate_grid_work,
)
from hydra_suite.core.inference.direct_calibration_sweep import MergeSettings
from hydra_suite.detectkit.jobs.direct_calibration import (
    EXHAUSTIVE_LABEL_WARNING,
    DirectCalibrationRequest,
    DirectCalibrationWorker,
    EvidenceSet,
    collect_evidence,
    save_direct_calibration,
)
from hydra_suite.widgets.dialogs import BaseDialog

_TABLE_COLUMNS = [
    ("label", "Candidate"),
    ("tiles_per_frame", "Tiles/frame"),
    ("total_tiles", "Total tiles"),
    ("estimated_duration", "Estimated duration"),
    ("failed_reason", "Failure reason"),
]

# Same grid shape as core/inference/semantic/calibration.py's CONFIDENCE_GRID
# (0.05 to 0.95 in steps of 0.05, rounded to 2 decimals) -- deliberately a
# separate constant, not an import, so the direct-calibration path stays
# uncoupled from the semantic/SAM3 path. 0.05 is also the floor the parts
# are collected at, so it must stay the lowest value.
CONFIDENCE_GRID: tuple[float, ...] = tuple(
    round(0.05 + 0.05 * step, 2) for step in range(19)
)
assert CONFIDENCE_GRID[0] == 0.05 and CONFIDENCE_GRID[-1] == 0.95

# Default merge policy/metric swept across these thresholds -- so the
# "confidence x merge" label is honest about there being an actual grid.
MERGE_THRESHOLD_GRID: tuple[float, ...] = (0.3, 0.5, 0.7)

RUNTIME_TIERS: tuple[str, ...] = ("cpu", "gpu", "gpu_fast")


def _humanise_tiles(total_tiles: int) -> str:
    """Rough wall-clock proxy from tile count alone (no timing exists yet)."""
    if total_tiles <= 0:
        return "-"
    return f"{total_tiles} tile-passes"


class DirectCalibrationWizard(BaseDialog):
    """Gate dialog: evidence summary, candidate grid + cost, then affirm-and-run."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        model_path,
        task: str,
        dataset_yaml,
        sources: list,
        training_geometry: dict,
        evidence_dir,
        split: str = "val",
        budget: int = 80,
        max_total_tiles: int = DEFAULT_MAX_TOTAL_TILES,
        runtime_tier: str = "gpu",
    ) -> None:
        super().__init__(
            "SAHI calibration — Experimental calibration",
            parent,
            buttons=QDialogButtonBox.Cancel,
        )
        self._model_path = Path(model_path)
        self._task = task
        self._training_geometry = dict(training_geometry)
        self._evidence_dir = Path(evidence_dir)
        self._max_total_tiles = int(max_total_tiles)
        self._default_runtime_tier = (
            runtime_tier if runtime_tier in RUNTIME_TIERS else "gpu"
        )

        self._evidence = collect_evidence(
            dataset_yaml=dataset_yaml, sources=sources, split=split, budget=budget
        )
        self.candidates = build_candidate_grid(self._training_geometry)
        frame_hw = self._evidence.size_range[1]
        imgsz = int(self._training_geometry.get("imgsz") or 640)
        self.estimates = estimate_grid_work(
            self.candidates,
            frame_hw=frame_hw,
            imgsz=imgsz,
            frames=len(self._evidence.frames),
            max_total_tiles=self._max_total_tiles,
        )

        self._build_ui()
        self._populate_summary()
        self._populate_table()

        # Run button lives outside the standard box: it needs custom gating
        # (exhaustive-labels + broad-sweep confirmation + non-empty evidence),
        # not plain accept.
        self.btn_run = self._buttons.addButton(
            "Run calibration", QDialogButtonBox.AcceptRole
        )
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.accept)

        if not self._evidence.frames or not self._evidence.instances:
            self.set_calibration_enabled(
                False,
                "No labelled evidence frames were found -- calibration "
                "cannot run against zero evidence.",
            )
        else:
            self._update_run_enabled()

    def _build_ui(self) -> None:
        self.lbl_evidence_summary = QLabel()
        self.lbl_evidence_summary.setWordWrap(True)
        self.add_content(self.lbl_evidence_summary)

        largest_per_frame = 0
        for _path, labels in self._evidence.frames:
            largest_per_frame = max(largest_per_frame, len(labels))
        default_max_targets = max(20, largest_per_frame)

        self.spin_max_targets = QSpinBox()
        self.spin_max_targets.setRange(1, 100000)
        self.spin_max_targets.setValue(default_max_targets)
        self.spin_max_targets.setToolTip(
            "Maximum detections kept per frame before merging. A value below "
            "the real animal count caps recall no matter how good the "
            "detector is -- this is the constraint that makes a measurement "
            "honest."
        )
        max_targets_label = QLabel("Max targets per frame:")
        max_targets_box = QWidget()
        max_targets_layout = QVBoxLayout(max_targets_box)
        max_targets_layout.setContentsMargins(0, 0, 0, 0)
        max_targets_layout.addWidget(max_targets_label)
        max_targets_layout.addWidget(self.spin_max_targets)
        self.add_content(max_targets_box)

        self.table_candidates = QTableWidget(0, len(_TABLE_COLUMNS))
        self.table_candidates.setHorizontalHeaderLabels(
            [label for _key, label in _TABLE_COLUMNS]
        )
        self.table_candidates.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.table_candidates.setEditTriggers(QTableWidget.NoEditTriggers)
        self.add_content(self.table_candidates)

        default_merge = MergeSettings()
        row_count = (
            len(self.candidates) * len(CONFIDENCE_GRID) * len(MERGE_THRESHOLD_GRID)
        )
        confidence_group = QGroupBox("Sweep grid (confidence x merge)")
        confidence_layout = QVBoxLayout(confidence_group)
        confidence_layout.addWidget(
            QLabel(
                f"Confidence grid: {CONFIDENCE_GRID[0]:g}-{CONFIDENCE_GRID[-1]:g} "
                f"step 0.05 ({len(CONFIDENCE_GRID)} values). Merge policy: "
                f"{default_merge.policy} · metric: {default_merge.metric} · "
                "thresholds: " + ", ".join(f"{t:g}" for t in MERGE_THRESHOLD_GRID) + "."
            )
        )
        confidence_layout.addWidget(
            QLabel(
                f"This measures {len(self.candidates)} candidates x "
                f"{len(CONFIDENCE_GRID)} confidences x {len(MERGE_THRESHOLD_GRID)} "
                f"merge settings = {row_count} rows, all replayed offline from "
                "one model pass per candidate at zero extra model cost."
            )
        )
        self.add_content(confidence_group)

        runtime_label = QLabel("Runtime tier measured (affects s/frame shown):")
        self.combo_runtime_tier = QComboBox()
        self.combo_runtime_tier.addItems(list(RUNTIME_TIERS))
        self.combo_runtime_tier.setCurrentText(self._default_runtime_tier)
        self.combo_runtime_tier.setToolTip(
            "The measured seconds/frame is only honest if it is measured on "
            "the tier the project actually tracks with."
        )
        runtime_box = QWidget()
        runtime_layout = QVBoxLayout(runtime_box)
        runtime_layout.setContentsMargins(0, 0, 0, 0)
        runtime_layout.addWidget(runtime_label)
        runtime_layout.addWidget(self.combo_runtime_tier)
        self.add_content(runtime_box)

        rule_label = QLabel(f"Recommendation rule: {RECOMMENDATION_RULE}")
        rule_label.setWordWrap(True)
        self.add_content(rule_label)

        self.chk_exhaustive = QCheckBox(EXHAUSTIVE_LABEL_WARNING)
        self.chk_exhaustive.setChecked(False)
        self.chk_exhaustive.toggled.connect(self._update_run_enabled)
        self.add_content(self.chk_exhaustive)

        self.chk_confirm_broad_sweep = QCheckBox(
            "Confirm running a broader sweep despite candidates that exceed "
            f"the {self._max_total_tiles}-tile budget (may take hours)."
        )
        self.chk_confirm_broad_sweep.setChecked(False)
        self.chk_confirm_broad_sweep.toggled.connect(self._update_run_enabled)
        self.add_content(self.chk_confirm_broad_sweep)

    def _populate_summary(self) -> None:
        evidence = self._evidence
        low_hw, high_hw = evidence.size_range
        split_note = ""
        if evidence.split not in ("val", "sources"):
            split_note = (
                f" (fell back to the '{evidence.split}' split -- no labelled "
                "'val' frames were found; never trust this silently)"
            )
        elif evidence.split == "sources":
            split_note = " (drawn from raw sources; no dataset split was labelled)"
        text = (
            f"{len(evidence.frames)} frames, {evidence.instances} instances, "
            f"sampled from {evidence.sampled_from} candidate frames "
            f"(cap {len(evidence.frames)} of {evidence.sampled_from}). "
            f"Image size range: {low_hw[1]}x{low_hw[0]} to "
            f"{high_hw[1]}x{high_hw[0]}. Split: {evidence.split}{split_note}."
        )
        self.lbl_evidence_summary.setText(text)

    def _populate_table(self) -> None:
        self.table_candidates.setRowCount(len(self.estimates))
        for row, estimate in enumerate(self.estimates):
            values = [
                estimate.candidate.label,
                str(estimate.tiles_per_frame),
                str(estimate.total_tiles),
                _humanise_tiles(estimate.total_tiles),
                estimate.failed_reason,
            ]
            for col, value in enumerate(values):
                self.table_candidates.setItem(row, col, QTableWidgetItem(value))

    def _any_over_budget(self) -> bool:
        return any(estimate.failed_reason for estimate in self.estimates)

    def _update_run_enabled(self) -> None:
        if not hasattr(self, "btn_run"):
            return
        if not self._evidence.frames or not self._evidence.instances:
            self.btn_run.setEnabled(False)
            return
        enabled = self.chk_exhaustive.isChecked()
        if self._any_over_budget() and not self.chk_confirm_broad_sweep.isChecked():
            enabled = False
        self.btn_run.setEnabled(enabled)

    def set_calibration_enabled(self, enabled: bool, reason: str = "") -> None:
        """External gate (e.g. no model loaded yet); overrides internal state."""
        self.chk_exhaustive.setEnabled(enabled)
        self.chk_confirm_broad_sweep.setEnabled(enabled)
        if not enabled:
            self.btn_run.setEnabled(False)
            if reason:
                self.btn_run.setToolTip(reason)
                self.lbl_evidence_summary.setText(
                    self.lbl_evidence_summary.text() + f"\n\n{reason}"
                )
        else:
            self.btn_run.setToolTip("")
            self._update_run_enabled()

    def evidence(self) -> EvidenceSet:
        return self._evidence

    def request(self) -> DirectCalibrationRequest:
        default_merge = MergeSettings()
        merge_settings = tuple(
            MergeSettings(
                policy=default_merge.policy,
                metric=default_merge.metric,
                threshold=threshold,
            )
            for threshold in MERGE_THRESHOLD_GRID
        )
        return DirectCalibrationRequest(
            model_path=self._model_path,
            task=self._task,
            evidence=self._evidence,
            candidates=self.candidates,
            confidences=CONFIDENCE_GRID,
            merge_settings=merge_settings,
            runtime_tier=self.combo_runtime_tier.currentText(),
            max_targets=int(self.spin_max_targets.value()),
            evidence_dir=self._evidence_dir,
        )


def open_direct_calibration(
    parent,
    *,
    model_path,
    task: str,
    dataset_yaml,
    sources: list,
    training_geometry: dict,
    evidence_dir,
    runtime_tier: str = "gpu",
) -> list[dict]:
    """The single launcher every entry point calls: wizard -> worker -> results.

    Returns the profiles the user chose to save (``staged_profiles()`` from
    ``DirectCalibrationResultsDialog``), or ``[]`` if the user cancelled at
    any stage (the wizard's gate, the sweep itself, or the results dialog).
    A partial (cancelled-mid-sweep) outcome is still shown for inspection in
    the results dialog, but ``save_direct_calibration`` -- called before the
    dialog opens, exactly as it is everywhere else in this feature -- never
    lets a partial run overwrite previously saved complete evidence.
    """
    # Local import: avoids a training_dialog.py <-> direct_calibration_results
    # cycle at module load time (results dialog is only needed once the
    # sweep has actually produced an outcome to show).
    from .direct_calibration_results import DirectCalibrationResultsDialog

    wizard = DirectCalibrationWizard(
        parent,
        model_path=model_path,
        task=task,
        dataset_yaml=dataset_yaml,
        sources=sources,
        training_geometry=training_geometry,
        evidence_dir=evidence_dir,
        runtime_tier=runtime_tier,
    )
    if wizard.exec() != QDialog.DialogCode.Accepted:
        return []

    request = wizard.request()

    progress = QProgressDialog("Measuring SAHI candidates…", "Cancel", 0, 100, parent)
    progress.setWindowTitle("SAHI calibration")
    progress.setMinimumDuration(0)
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    progress.setMinimumWidth(420)

    worker = DirectCalibrationWorker(request)
    progress.canceled.connect(worker.cancel)
    worker.progress.connect(progress.setValue)
    worker.status.connect(progress.setLabelText)

    loop = QEventLoop()
    outcome_holder: dict = {}
    worker.result_ready.connect(
        lambda outcome: outcome_holder.__setitem__("outcome", outcome)
    )
    worker.finished.connect(loop.quit)

    progress.show()
    progress.raise_()
    progress.activateWindow()
    worker.start()
    loop.exec()
    progress.close()

    outcome = outcome_holder.get("outcome")
    if outcome is None:
        return []
    if not outcome.points:
        QMessageBox.information(
            parent,
            "SAHI calibration",
            "Calibration produced nothing: it was cancelled before the "
            "first candidate finished measuring.",
        )
        return []

    # Persist BEFORE showing results: a partial run never overwrites complete
    # evidence -- save_direct_calibration enforces that half; this call is
    # what makes the enforcement reachable from the GUI at all.
    save_direct_calibration(Path(evidence_dir), outcome, request)

    results = DirectCalibrationResultsDialog(
        parent,
        model_path=model_path,
        outcome=outcome,
        training_geometry=training_geometry,
        previews=outcome.previews,
        task=task,
    )
    if outcome.partial:
        # Shown for inspection only -- clearly marked, never silently
        # treated as a complete measurement.
        results.setWindowTitle(
            "SAHI calibration results — PARTIAL (cancelled before completion)"
        )
    if results.exec() != QDialog.DialogCode.Accepted:
        return []
    return results.staged_profiles()
