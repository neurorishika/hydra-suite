"""Background-subtraction parameter auto-tuner dialog.

Mirrors the Tracking Auto-Tuner (``ParameterHelperDialog``) UI pattern:
domain-constraints banner, scoring-weights tab, parameter-checkbox tab,
styled control bar, colour-coded results table, and Apply / Cancel bottom bar.

Launches a :class:`BgSubtractionOptimizer` Optuna study, shows live progress,
and lets the user apply the best result back to the main window.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List

import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer, Slot
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.core.background.optimizer import (
    BgDetectionPreviewWorker,
    BgOptimizationResult,
    BgSubtractionOptimizer,
)
from hydra_suite.widgets.dialogs import BaseDialog

logger = logging.getLogger(__name__)

# ── Colour helpers ────────────────────────────────────────────────────────────


def _score_to_color(value: float) -> QColor:
    """Green (1.0) → red (0.0) colour mapping for a [0,1] score."""
    v = max(0.0, min(1.0, value))
    r = int((1.0 - v) * 200)
    g = int(v * 180)
    return QColor(r, g, 60, 220)


def _badge_item(score: float, display: str) -> QTableWidgetItem:
    """Colour-coded table cell; *score* in [0,1] where 1 = best (green)."""
    item = QTableWidgetItem(display)
    item.setBackground(QBrush(_score_to_color(score)))
    item.setForeground(QBrush(QColor(255, 255, 255)))
    f = QFont()
    f.setFamily("Monospace")
    f.setPointSize(8)
    item.setFont(f)
    item.setTextAlignment(Qt.AlignCenter)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item


def _plain_item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    f = QFont()
    f.setFamily("Monospace")
    f.setPointSize(8)
    item.setFont(f)
    item.setTextAlignment(Qt.AlignCenter)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item


# ── Dialog ────────────────────────────────────────────────────────────────────


class BgParameterHelperDialog(BaseDialog):
    """Modal dialog for Optuna BG-subtraction parameter optimisation."""

    def __init__(
        self,
        video_path: str,
        current_params: Dict[str, Any],
        parent: Any = None,
    ) -> None:
        super().__init__(
            title="Detection Auto-Tuner  (Background Subtraction)",
            parent=parent,
            buttons=QDialogButtonBox.NoButton,
            apply_dark_style=True,
        )
        self.setMinimumSize(1200, 640)

        self.video_path = video_path
        self.base_params = dict(current_params)
        self.results: List[BgOptimizationResult] = []
        self.optimizer: BgSubtractionOptimizer | None = None
        self.preview_worker: BgDetectionPreviewWorker | None = None

        # Preview state
        self._prev_panning = False
        self._prev_pan_start = None
        self._prev_scroll_h = 0
        self._prev_scroll_v = 0
        self._prev_frames: List[np.ndarray] = []
        self._prev_current_idx = 0
        self._prev_zoom_anchor: tuple[int, int, float, float] | None = None
        self._prev_auto_fit_pending = True

        self._build_ui()

    # ══════════════════════════════════════════════════════════════════════════
    # UI Construction
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self) -> None:
        container = QWidget()
        root = QVBoxLayout(container)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # ── Outer splitter: left (controls + table) | right (preview) ─────
        outer = QSplitter(Qt.Horizontal)
        outer.setChildrenCollapsible(False)

        # ── Left panel ────────────────────────────────────────────────────
        left_w = QWidget()
        left = QVBoxLayout(left_w)
        left.setSpacing(6)
        left.setContentsMargins(8, 8, 4, 8)

        # ── Domain constraints banner ─────────────────────────────────────
        left.addWidget(self._make_domain_banner())

        # ── Tabs: Scoring Weights | Optimization Parameters ──────────────
        tab = QTabWidget()
        tab.setStyleSheet("QTabBar::tab { padding: 5px 14px; }")
        tab.addTab(self._make_scoring_tab(), "\u2696  Scoring Weights")
        tab.addTab(self._make_params_tab(), "Optimization Parameters")
        left.addWidget(tab)

        # ── Optimization control bar ──────────────────────────────────────
        left.addLayout(self._make_opt_bar())

        # ── Progress ──────────────────────────────────────────────────────
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)
        left.addWidget(self.progress)

        self.status_label = QLabel("Select parameters above, then click Run.")
        self.status_label.setStyleSheet("color: #ffffff; font-size: 11px;")
        left.addWidget(self.status_label)

        # ── Results table ─────────────────────────────────────────────────
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            [
                "Rank",
                "Score",
                "Count\u2191",
                "Consist\u2191",
                "Stabil\u2191",
                "Med Area",
                "Threshold",
                "Key Changes",
            ]
        )
        hdr = self.table.horizontalHeader()
        hdr.setStretchLastSection(True)
        for col in range(7):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setDefaultSectionSize(36)
        left.addWidget(self.table, stretch=1)

        # ── Bottom bar ────────────────────────────────────────────────────
        left.addLayout(self._make_bottom_bar())

        outer.addWidget(left_w)

        # ── Right panel — preview ─────────────────────────────────────────
        outer.addWidget(self._make_preview_panel())

        # Give left ~60 % and preview ~40 % of initial width
        outer.setSizes([720, 480])

        root.addWidget(outer)
        self.add_content(container)

    # ── Domain constraints banner ─────────────────────────────────────────────

    def _make_domain_banner(self) -> QGroupBox:
        box = QGroupBox(
            "Physical Constraints  \u2014  read from Main Window "
            "(close & adjust there if needed)"
        )
        box.setStyleSheet(
            "QGroupBox { border: 1px solid #7a5f20; border-radius: 4px;"
            " margin-top: 6px; padding-top: 4px; color: #f0c060; font-weight: bold; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
        )
        lay = QHBoxLayout(box)
        lay.setContentsMargins(8, 4, 8, 4)

        _max_t = self.base_params.get("MAX_TARGETS", 5)
        _resize = self.base_params.get("RESIZE_FACTOR", 1.0)
        _start = self.base_params.get("START_FRAME", 0)
        _end = self.base_params.get("END_FRAME", "?")
        _dark = self.base_params.get("DARK_ON_LIGHT_BACKGROUND", True)
        _mode = "dark-on-light" if _dark else "light-on-dark"

        summary = QLabel(
            f"\u25cf\u00a0Max targets: <b>{_max_t}</b>"
            f"\u2002\u2502\u2002"
            f"\u25cf\u00a0Resize: <b>{_resize:.2f}</b>"
            f"\u2002\u2502\u2002"
            f"\u25cf\u00a0Mode: <b>{_mode}</b>"
            f"\u2002\u2502\u2002"
            f"\u25cf\u00a0Frames: <b>{_start}\u2013{_end}</b>"
        )
        summary.setTextFormat(Qt.RichText)
        summary.setStyleSheet("font-size: 11px; color: #ffffff; font-weight: normal;")
        summary.setToolTip(
            "These values come from the Main Window.\n"
            "MAX_TARGETS controls the count-accuracy objective.\n"
            "Close this dialog, change values, and reopen to use different constraints."
        )
        lay.addWidget(summary)
        lay.addStretch()
        return box

    # ── Scoring weights tab ───────────────────────────────────────────────────

    # (count, consistency, stability)
    _WEIGHT_PRESETS: Dict[str, tuple] = {
        "Balanced (default)": (0.50, 0.30, 0.20),
        "Count-first — many animals, erratic sizes": (0.70, 0.15, 0.15),
        "Consistency-first — uniform animals, few targets": (0.25, 0.50, 0.25),
        "Stability-first — steady tracking over time": (0.25, 0.25, 0.50),
    }

    def _make_scoring_tab(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setSpacing(8)

        note = QLabel(
            "Each weight controls how much that objective influences the composite "
            "score.  Weights are normalised at runtime \u2014 use any scale.\n"
            "Set a weight to <b>0</b> to disable that term entirely."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #ffffff; font-size: 11px;")
        outer.addWidget(note)

        # Preset selector
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self._combo_presets = QComboBox()
        for name in self._WEIGHT_PRESETS:
            self._combo_presets.addItem(name)
        self._combo_presets.activated.connect(
            lambda _: self._apply_weight_preset(self._combo_presets.currentText())
        )
        preset_row.addWidget(self._combo_presets, stretch=1)
        btn_ap = QPushButton("Apply")
        btn_ap.clicked.connect(
            lambda: self._apply_weight_preset(self._combo_presets.currentText())
        )
        preset_row.addWidget(btn_ap)
        outer.addLayout(preset_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3a3a;")
        outer.addWidget(sep)

        # Weight spinboxes
        box = QGroupBox("Objective Weights")
        grid = QGridLayout(box)
        grid.setVerticalSpacing(10)
        grid.setHorizontalSpacing(10)
        grid.setColumnStretch(2, 1)

        def _spin(default: float, tip: str) -> QDoubleSpinBox:
            s = QDoubleSpinBox()
            s.setRange(0.0, 10.0)
            s.setSingleStep(0.05)
            s.setDecimals(2)
            s.setValue(default)
            s.setFixedWidth(78)
            s.setToolTip(tip)
            s.valueChanged.connect(self._update_weight_sum)
            return s

        def _hint(up: str, down: str) -> QLabel:
            lbl = QLabel(
                f"<span style='color:#6ec07a'>\u2191</span> "
                f"<span style='color:#ffffff'>{up}</span>"
                f"<br><span style='color:#e0875a'>\u2193</span> "
                f"<span style='color:#ffffff'>{down}</span>"
            )
            lbl.setWordWrap(True)
            lbl.setTextFormat(Qt.RichText)
            lbl.setStyleSheet("font-size: 10px;")
            return lbl

        self.spin_w_count = _spin(
            0.50,
            "Fraction of frames where detection count == MAX_TARGETS.\n"
            "Primary objective for reliable detection.",
        )
        self.spin_w_consistency = _spin(
            0.30,
            "How uniform are detection sizes within a single frame?\n"
            "(1 \u2212 coefficient of variation).\n"
            "Higher = animals should be similarly-sized.",
        )
        self.spin_w_stability = _spin(
            0.20,
            "How stable is the median detection area across frames?\n"
            "(1 \u2212 CoV of per-frame medians).\n"
            "Higher = sizes should not fluctuate over time.",
        )

        rows = [
            (
                "Count",
                self.spin_w_count,
                _hint(
                    "Detections are lost or split \u2014 prioritise correct count",
                    "Count is already reliable \u2014 favour size quality",
                ),
            ),
            (
                "Consistency",
                self.spin_w_consistency,
                _hint(
                    "Animals are similar-sized \u2014 penalise size variation",
                    "Mixed sizes are natural (e.g.\u00a0larvae vs adults)",
                ),
            ),
            (
                "Stability",
                self.spin_w_stability,
                _hint(
                    "Detection sizes should stay stable across video",
                    "Size changes through video are expected (growth, movement)",
                ),
            ),
        ]
        for row_idx, (lbl_text, spin, hint_lbl) in enumerate(rows):
            name_lbl = QLabel(f"<b>{lbl_text}</b>")
            name_lbl.setFixedWidth(110)
            grid.addWidget(name_lbl, row_idx, 0)
            grid.addWidget(spin, row_idx, 1)
            grid.addWidget(hint_lbl, row_idx, 2)

        outer.addWidget(box)

        # Footer: sum + renormalize + reset
        ctrl = QHBoxLayout()
        self.lbl_weight_sum = QLabel("Sum: 1.00")
        self.lbl_weight_sum.setStyleSheet("font-size: 11px; color: #9cdcfe;")
        ctrl.addWidget(self.lbl_weight_sum)
        ctrl.addStretch(1)

        btn_norm = QPushButton("Renormalize")
        btn_norm.setToolTip("Scale weights proportionally to sum to 1.0.")
        btn_norm.clicked.connect(self._renormalize_weights)
        ctrl.addWidget(btn_norm)

        btn_reset = QPushButton("Reset to Defaults")
        btn_reset.clicked.connect(self._reset_weights)
        ctrl.addWidget(btn_reset)
        outer.addLayout(ctrl)

        outer.addStretch()
        return w

    # ── Weight helpers ────────────────────────────────────────────────────────

    def _weight_spins(self):
        return [self.spin_w_count, self.spin_w_consistency, self.spin_w_stability]

    def _apply_weight_preset(self, name: str):
        preset = self._WEIGHT_PRESETS.get(name)
        if preset is None:
            return
        for spin, val in zip(self._weight_spins(), preset):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
        self._update_weight_sum()

    def _renormalize_weights(self):
        spins = self._weight_spins()
        total = sum(s.value() for s in spins)
        if total < 1e-9:
            return
        for s in spins:
            s.blockSignals(True)
            s.setValue(round(s.value() / total, 4))
            s.blockSignals(False)
        self._update_weight_sum()

    def _update_weight_sum(self):
        total = sum(s.value() for s in self._weight_spins())
        color = "#9cdcfe" if abs(total - 1.0) < 0.01 else "#f0a050"
        self.lbl_weight_sum.setText(f"Sum: {total:.2f}")
        self.lbl_weight_sum.setStyleSheet(f"font-size: 11px; color: {color};")

    def _reset_weights(self):
        self._apply_weight_preset("Balanced (default)")

    def get_scoring_weights(self) -> Dict[str, float]:
        """Return the current COUNT, CONSISTENCY, and STABILITY objective weights as a param dict."""
        return {
            "SCORE_WEIGHT_COUNT": self.spin_w_count.value(),
            "SCORE_WEIGHT_CONSISTENCY": self.spin_w_consistency.value(),
            "SCORE_WEIGHT_STABILITY": self.spin_w_stability.value(),
        }

    # ── Parameters (checkboxes) tab ──────────────────────────────────────────

    def _make_params_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setSpacing(8)
        lay.setContentsMargins(4, 4, 4, 4)

        self._param_checkboxes: Dict[str, QCheckBox] = {}
        core_keys = {
            "THRESHOLD_VALUE",
            "MORPH_KERNEL_SIZE",
            "MIN_CONTOUR_AREA",
            "ENABLE_ADDITIONAL_DILATION",
            "DILATION_KERNEL_SIZE",
            "DILATION_ITERATIONS",
            "ENABLE_CONSERVATIVE_SPLIT",
            "CONSERVATIVE_KERNEL_SIZE",
            "CONSERVATIVE_ERODE_ITER",
        }
        self._core_param_keys = set(core_keys)

        note = QLabel(
            "Select any background-subtraction factors to include in the search. "
            "When an enable flag and its child parameters are both selected, the child "
            "values are still explored so the search space stays stable instead of being "
            "conditionally collapsed."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #ffffff; font-size: 11px;")
        lay.addWidget(note)

        quick_row = QHBoxLayout()
        btn_select_all = QPushButton("Select All")
        btn_select_all.clicked.connect(lambda: self._set_param_selection(None))
        quick_row.addWidget(btn_select_all)
        btn_core = QPushButton("Core Only")
        btn_core.clicked.connect(
            lambda: self._set_param_selection(self._core_param_keys)
        )
        quick_row.addWidget(btn_core)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(lambda: self._set_param_selection(set()))
        quick_row.addWidget(btn_clear)
        quick_row.addStretch(1)
        lay.addLayout(quick_row)

        def _cb(name: str, checked: bool = False) -> QCheckBox:
            cb = QCheckBox(name)
            cb.setChecked(checked)
            self._param_checkboxes[name] = cb
            return cb

        # --- Image adjustments ---
        lay.addWidget(
            self._checkboxes_in_group(
                "Image & Polarity",
                [
                    (
                        _cb("BRIGHTNESS"),
                        "Brightness adjustment applied before subtraction.",
                    ),
                    (
                        _cb("CONTRAST"),
                        "Contrast multiplier applied before subtraction.",
                    ),
                    (_cb("GAMMA"), "Gamma correction applied before subtraction."),
                    (
                        _cb("DARK_ON_LIGHT_BACKGROUND"),
                        "Whether animals are darker than the background.",
                    ),
                ],
            )
        )

        # --- Background model group ---
        lay.addWidget(
            self._checkboxes_in_group(
                "Background Model",
                [
                    (
                        _cb("BACKGROUND_PRIME_FRAMES"),
                        "How many frames are used to build the initial lightest-pixel background.",
                    ),
                    (
                        _cb("ENABLE_ADAPTIVE_BACKGROUND"),
                        "Whether the background updates over time.",
                    ),
                    (
                        _cb("BACKGROUND_LEARNING_RATE"),
                        "Adaptive background learning rate.",
                    ),
                ],
            )
        )

        # --- Lighting group ---
        lay.addWidget(
            self._checkboxes_in_group(
                "Lighting Stabilization",
                [
                    (
                        _cb("ENABLE_LIGHTING_STABILIZATION"),
                        "Enable temporal lighting normalization.",
                    ),
                    (
                        _cb("LIGHTING_SMOOTH_FACTOR"),
                        "Temporal smoothing factor for lighting stabilization.",
                    ),
                    (
                        _cb("LIGHTING_MEDIAN_WINDOW"),
                        "Median window for lighting estimation (odd).",
                    ),
                ],
            )
        )

        # --- Thresholding group ---
        lay.addWidget(
            self._checkboxes_in_group(
                "Thresholding & Contours",
                [
                    (
                        _cb("THRESHOLD_VALUE", checked=True),
                        "Binary threshold applied to the difference image.",
                    ),
                    (
                        _cb("MORPH_KERNEL_SIZE", checked=True),
                        "Morphological open/close kernel size (odd).",
                    ),
                    (
                        _cb("MIN_CONTOUR_AREA", checked=True),
                        "Minimum contour area in pixels.",
                    ),
                    (
                        _cb("MAX_CONTOUR_MULTIPLIER"),
                        "Maximum contour count multiplier before the frame is rejected as noisy.",
                    ),
                ],
            )
        )

        lay.addWidget(
            self._checkboxes_in_group(
                "Size Filtering",
                [
                    (
                        _cb("ENABLE_SIZE_FILTERING"),
                        "Enable filtering against expected one-animal size bounds.",
                    ),
                    (
                        _cb("MIN_OBJECT_SIZE"),
                        "Minimum object size multiplier relative to the current body-size estimate.",
                    ),
                    (
                        _cb("MAX_OBJECT_SIZE"),
                        "Maximum object size multiplier relative to the current body-size estimate.",
                    ),
                ],
            )
        )

        # --- Dilation group ---
        lay.addWidget(
            self._checkboxes_in_group(
                "Additional Dilation",
                [
                    (
                        _cb("ENABLE_ADDITIONAL_DILATION", checked=True),
                        "Toggle additional dilation pass.",
                    ),
                    (
                        _cb("DILATION_KERNEL_SIZE", checked=True),
                        "Dilation kernel size (odd).",
                    ),
                    (
                        _cb("DILATION_ITERATIONS", checked=True),
                        "Number of dilation iterations.",
                    ),
                ],
            )
        )

        # --- Conservative split group ---
        lay.addWidget(
            self._checkboxes_in_group(
                "Conservative Split",
                [
                    (
                        _cb("ENABLE_CONSERVATIVE_SPLIT", checked=True),
                        "Toggle local re-thresholding split.",
                    ),
                    (
                        _cb("CONSERVATIVE_KERNEL_SIZE", checked=True),
                        "Kernel size for erosion before re-thresholding.",
                    ),
                    (
                        _cb("CONSERVATIVE_ERODE_ITER", checked=True),
                        "Number of erosion iterations.",
                    ),
                ],
            )
        )

        lay.addStretch()
        scroll.setWidget(container)
        return scroll

    def _set_param_selection(self, selected_keys: set[str] | None) -> None:
        """Set checkbox states for all tunable parameters."""
        for key, checkbox in self._param_checkboxes.items():
            checkbox.setChecked(True if selected_keys is None else key in selected_keys)

    @staticmethod
    def _checkboxes_in_group(
        title: str,
        items: list,
    ) -> QGroupBox:
        box = QGroupBox(title)
        grid = QGridLayout(box)
        grid.setSpacing(6)
        for idx, (cb, tooltip) in enumerate(items):
            if tooltip:
                cb.setToolTip(tooltip)
            grid.addWidget(cb, idx // 2, idx % 2)
        return box

    # ── Optimization control bar ──────────────────────────────────────────────

    def _make_opt_bar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        row.addWidget(QLabel("Trials:"))
        self.spin_trials = QSpinBox()
        self.spin_trials.setRange(10, 500)
        self.spin_trials.setValue(50)
        self.spin_trials.setToolTip(
            "Number of Bayesian optimisation trials (parameter combinations)."
        )
        row.addWidget(self.spin_trials)

        row.addWidget(QLabel("Sample frames:"))
        self.spin_frames = QSpinBox()
        self.spin_frames.setRange(5, 200)
        self.spin_frames.setValue(30)
        self.spin_frames.setToolTip(
            "Number of evenly-spaced frames to sample.\n"
            "More frames \u2192 more accurate but slower."
        )
        row.addWidget(self.spin_frames)

        row.addWidget(QLabel("Sampler:"))
        self.combo_sampler = QComboBox()
        self.combo_sampler.addItems(
            [
                "Auto (recommended)",
                "GP \u2014 best \u2264200 trials",
                "TPE \u2014 best >500 trials",
            ]
        )
        self.combo_sampler.setCurrentIndex(2)  # TPE default for detection
        self.combo_sampler.setToolTip(
            "Bayesian optimisation algorithm.\n"
            "Auto: OptunaHub AutoSampler \u2014 GP early on, TPE later. Best overall.\n"
            "GP:   Gaussian Process (Mat\u00e9rn-2.5, ARD, log-EI). Fastest convergence\n"
            "      for 50\u2013200 trials; requires scipy + torch.\n"
            "TPE:  Multivariate Tree-structured Parzen Estimator. Robust fallback;\n"
            "      no extra dependencies, scales well beyond 500 trials."
        )
        row.addWidget(self.combo_sampler)

        row.addStretch(1)

        self.btn_run = QPushButton("\u25b6  Run Bayesian Optimizer")
        self.btn_run.clicked.connect(self._on_run)
        self.btn_run.setStyleSheet(
            "background-color: #0e639c; color: white; "
            "font-weight: bold; padding: 8px 16px;"
        )
        row.addWidget(self.btn_run)

        self.btn_stop = QPushButton("\u25a0  Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setStyleSheet(
            "background-color: #c0392b; color: white; "
            "font-weight: bold; padding: 8px 12px;"
        )
        row.addWidget(self.btn_stop)

        return row

    # ── Bottom bar ────────────────────────────────────────────────────────────

    def _make_bottom_bar(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self.btn_preview = QPushButton("\u25b6  Preview Selected")
        self.btn_preview.setEnabled(False)
        self.btn_preview.setToolTip(
            "Run the selected parameter set on the sample frames\n"
            "and show detection overlays in the preview panel."
        )
        self.btn_preview.clicked.connect(self._run_preview)
        row.addWidget(self.btn_preview)

        self.btn_apply = QPushButton("\u2714  Apply Selected")
        self.btn_apply.setEnabled(False)
        self.btn_apply.setToolTip(
            "Write the selected parameter set back into the MAT setup tab."
        )
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_apply.setStyleSheet(
            "font-weight: bold; background-color: #28a745; "
            "color: white; padding: 8px 16px;"
        )
        row.addWidget(self.btn_apply)

        row.addStretch(1)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        row.addWidget(btn_cancel)

        return row

    # ══════════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════════

    def get_tuning_config(self) -> Dict[str, bool]:
        """Return a mapping of background-subtraction parameter names to whether each is enabled for auto-tuning."""
        return {
            key: checkbox.isChecked()
            for key, checkbox in self._param_checkboxes.items()
        }

    def get_selected_params(self) -> Dict[str, Any]:
        """Return the parameter dict of the user-selected trial."""
        row = getattr(self, "_selected_row", 0)
        if 0 <= row < len(self.results):
            return self.results[row].params
        return {}

    # ══════════════════════════════════════════════════════════════════════════
    # Slots
    # ══════════════════════════════════════════════════════════════════════════

    def _sampler_type(self) -> str:
        """Return sampler key from the combo box selection."""
        idx = self.combo_sampler.currentIndex()
        return ("auto", "gp", "tpe")[idx]

    @Slot()
    def _on_run(self) -> None:
        config = self.get_tuning_config()
        if not any(config.values()):
            QMessageBox.warning(
                self,
                "Selection Required",
                "Please select at least one parameter to tune.",
            )
            return

        self.results.clear()
        self.table.setRowCount(0)
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_apply.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)

        self.optimizer = BgSubtractionOptimizer(
            video_path=self.video_path,
            base_params=self.base_params,
            tuning_config=config,
            scoring_weights=self.get_scoring_weights(),
            n_trials=self.spin_trials.value(),
            n_sample_frames=self.spin_frames.value(),
            sampler_type=self._sampler_type(),
            parent=self,
        )
        self.optimizer.progress_signal.connect(self._on_progress)
        self.optimizer.result_signal.connect(self._on_results)
        self.optimizer.finished_signal.connect(self._on_finished)
        self.optimizer.start()

    @Slot()
    def _on_stop(self) -> None:
        if self.optimizer is not None:
            self.optimizer.stop()

    @Slot(int, str)
    def _on_progress(self, pct: int, msg: str) -> None:
        self.progress.setValue(pct)
        self.status_label.setText(msg)

    @Slot(list)
    def _on_results(self, results: list) -> None:
        self.results = results
        self._populate_table()

    @Slot()
    def _on_finished(self) -> None:
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setVisible(False)
        has_results = len(self.results) > 0
        self.btn_apply.setEnabled(has_results)
        self.btn_preview.setEnabled(has_results)
        if self.results:
            self.table.selectRow(0)
            n = len(self.results)
            best = self.results[0]
            self.status_label.setText(
                f"Done. {n} trials.  Best score: {best.score:.3f}  "
                f"(median area: {best.median_area:.0f} px\u00b2)"
            )
        else:
            self.status_label.setText("Search finished with no results.")

    @Slot()
    def _on_apply(self) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.results):
            QMessageBox.warning(self, "No Selection", "Select a row first.")
            return
        self._selected_row = row
        self.accept()

    # ── Table population ──────────────────────────────────────────────────────

    def _populate_table(self) -> None:
        self.table.setRowCount(0)
        for rank, r in enumerate(self.results[:50]):
            row = self.table.rowCount()
            self.table.insertRow(row)

            # Rank
            self.table.setItem(row, 0, _plain_item(str(rank + 1)))

            # Composite score
            self.table.setItem(row, 1, _badge_item(r.score, f"{r.score:.3f}"))

            # Sub-scores
            ss = r.sub_scores
            self.table.setItem(
                row,
                2,
                _badge_item(ss.get("count", 0), f"{ss.get('count', 0):.2f}"),
            )
            self.table.setItem(
                row,
                3,
                _badge_item(
                    ss.get("consistency", 0),
                    f"{ss.get('consistency', 0):.2f}",
                ),
            )
            self.table.setItem(
                row,
                4,
                _badge_item(
                    ss.get("stability", 0),
                    f"{ss.get('stability', 0):.2f}",
                ),
            )

            # Median area
            self.table.setItem(
                row,
                5,
                _plain_item(f"{r.median_area:.0f}"),
            )

            # Threshold
            self.table.setItem(
                row,
                6,
                _plain_item(str(r.params.get("THRESHOLD_VALUE", "?"))),
            )

            # Key changes
            changes = self._format_changes(r.params)
            chg_item = QTableWidgetItem(changes)
            chg_item.setFlags(chg_item.flags() & ~Qt.ItemIsEditable)
            chg_item.setToolTip(changes.replace("  ", "\n"))
            self.table.setItem(row, 7, chg_item)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_changes(self, params: Dict[str, Any]) -> str:
        """Compact one-liner: tuned value with \u25b2/\u25bc/= vs base params."""
        labels = {
            "BRIGHTNESS": ("Bri", "d"),
            "CONTRAST": ("Con", ".2f"),
            "GAMMA": ("Gam", ".2f"),
            "DARK_ON_LIGHT_BACKGROUND": ("Pol", ""),
            "BACKGROUND_PRIME_FRAMES": ("Prime", "d"),
            "ENABLE_ADAPTIVE_BACKGROUND": ("Adapt", ""),
            "BACKGROUND_LEARNING_RATE": ("Learn", ".4f"),
            "ENABLE_LIGHTING_STABILIZATION": ("Light", ""),
            "LIGHTING_SMOOTH_FACTOR": ("LSm", ".3f"),
            "LIGHTING_MEDIAN_WINDOW": ("LWin", "d"),
            "THRESHOLD_VALUE": ("Thr", "d"),
            "MORPH_KERNEL_SIZE": ("Morph", "d"),
            "MIN_CONTOUR_AREA": ("MinC", "d"),
            "MAX_CONTOUR_MULTIPLIER": ("MaxC", "d"),
            "ENABLE_SIZE_FILTERING": ("Size", ""),
            "MIN_OBJECT_SIZE": ("MinS", "d"),
            "MAX_OBJECT_SIZE": ("MaxS", "d"),
            "ENABLE_ADDITIONAL_DILATION": ("Dil", ""),
            "DILATION_KERNEL_SIZE": ("DilK", "d"),
            "DILATION_ITERATIONS": ("DilI", "d"),
            "ENABLE_CONSERVATIVE_SPLIT": ("Split", ""),
            "CONSERVATIVE_KERNEL_SIZE": ("SplK", "d"),
            "CONSERVATIVE_ERODE_ITER": ("SplE", "d"),
        }
        parts = []
        for key, (abbr, fmt) in labels.items():
            if key not in params:
                continue
            val = params[key]
            base_val = self.base_params.get(key)

            if isinstance(val, bool):
                val_str = "on" if val else "off"
            elif fmt:
                val_str = f"{val:{fmt}}"
            else:
                val_str = str(val)

            if base_val is not None:
                try:
                    diff = float(val) - float(base_val)
                    arrow = (
                        "\u25b2" if diff > 1e-9 else ("\u25bc" if diff < -1e-9 else "=")
                    )
                    parts.append(f"{abbr}:{val_str}{arrow}")
                except (TypeError, ValueError):
                    parts.append(f"{abbr}:{val_str}")
            else:
                parts.append(f"{abbr}:{val_str}")
        return "  ".join(parts)

    # ══════════════════════════════════════════════════════════════════════════
    # Preview Panel
    # ══════════════════════════════════════════════════════════════════════════

    def _make_preview_panel(self) -> QWidget:
        """Build the right-side preview pane: scroll area + zoom/frame sliders."""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 0, 0, 0)
        lay.setSpacing(4)

        title = QLabel("Detection Preview")
        title.setStyleSheet("color: #9cdcfe; font-weight: bold; font-size: 11px;")
        title.setAlignment(Qt.AlignCenter)
        lay.addWidget(title)

        self._prev_scroll = QScrollArea()
        self._prev_scroll.setWidgetResizable(False)
        self._prev_scroll.setAlignment(Qt.AlignCenter)
        self._prev_scroll.setStyleSheet(
            "background: #121212; border: 1px solid #3a3a3a;"
        )

        self._prev_label = QLabel(
            "No preview yet.\nRun optimizer, select a result row,\n"
            "then click \u25b6 Preview Selected."
        )
        self._prev_label.setAlignment(Qt.AlignCenter)
        self._prev_label.setStyleSheet("color: #ffffff; font-size: 12px;")
        self._prev_label.setMinimumSize(200, 150)
        self._prev_label.setMouseTracking(True)
        self._prev_label.mousePressEvent = self._on_prev_mouse_press
        self._prev_label.mouseMoveEvent = self._on_prev_mouse_move
        self._prev_label.mouseReleaseEvent = self._on_prev_mouse_release
        self._prev_label.mouseDoubleClickEvent = lambda _: self._fit_preview()
        self._prev_label.wheelEvent = self._on_prev_wheel
        self._prev_scroll.setWidget(self._prev_label)
        for target in (self._prev_label, self._prev_scroll.viewport()):
            target.setAttribute(Qt.WA_AcceptTouchEvents, True)
            target.grabGesture(Qt.PinchGesture)
            target.installEventFilter(self)
        lay.addWidget(self._prev_scroll, stretch=1)

        # Frame navigation slider
        frame_row = QHBoxLayout()
        frame_row.setSpacing(6)
        frame_lbl = QLabel("Frame:")
        frame_lbl.setFixedWidth(42)
        frame_row.addWidget(frame_lbl)

        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.setValue(0)
        self._frame_slider.setTracking(not sys.platform.startswith("linux"))
        self._frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        self._frame_slider.sliderMoved.connect(self._on_frame_slider_moved)
        frame_row.addWidget(self._frame_slider, stretch=1)

        self._frame_label = QLabel("0/0")
        self._frame_label.setFixedWidth(50)
        self._frame_label.setStyleSheet("font-size: 10px;")
        frame_row.addWidget(self._frame_label)
        lay.addLayout(frame_row)

        # Zoom slider
        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(6)
        zoom_lbl = QLabel("Zoom:")
        zoom_lbl.setFixedWidth(42)
        zoom_row.addWidget(zoom_lbl)

        self._prev_zoom_slider = QSlider(Qt.Horizontal)
        self._prev_zoom_slider.setRange(10, 400)
        self._prev_zoom_slider.setValue(100)
        self._prev_zoom_slider.setSingleStep(10)
        self._prev_zoom_slider.setTickPosition(QSlider.TicksBelow)
        self._prev_zoom_slider.setTickInterval(50)
        self._prev_zoom_slider.valueChanged.connect(self._on_prev_zoom_changed)
        zoom_row.addWidget(self._prev_zoom_slider, stretch=1)

        self._prev_zoom_label = QLabel("100%")
        self._prev_zoom_label.setFixedWidth(40)
        self._prev_zoom_label.setStyleSheet("font-size: 10px;")
        zoom_row.addWidget(self._prev_zoom_label)

        btn_fit = QPushButton("Fit")
        btn_fit.setFixedWidth(36)
        btn_fit.setToolTip(
            "Fit preview to available panel space (double-click also works)"
        )
        btn_fit.clicked.connect(self._fit_preview)
        zoom_row.addWidget(btn_fit)
        lay.addLayout(zoom_row)

        hint = QLabel("Ctrl+wheel: zoom  \u2022  drag: pan  \u2022  double-click: fit")
        hint.setStyleSheet("color: #ffffff; font-size: 9px;")
        hint.setAlignment(Qt.AlignCenter)
        lay.addWidget(hint)

        return w

    # ── Preview actions ───────────────────────────────────────────────────────

    def _preview_cache_kwargs(self) -> Dict[str, Any]:
        """Expose optimizer-cached raw frames so preview reuses the same sample set."""
        if self.optimizer is None:
            return {}
        return {
            "cached_prime_frames": getattr(
                self.optimizer, "_cached_prime_frames", None
            ),
            "cached_sample_frames": getattr(
                self.optimizer, "_cached_sample_frames", None
            ),
            "cached_sample_indices": getattr(
                self.optimizer, "_cached_sample_indices", None
            ),
            "roi_mask": getattr(self.optimizer, "_cached_roi_mask", None),
        }

    @Slot()
    def _run_preview(self) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.results):
            return
        res = self.results[row]
        trial_params = dict(res.params)

        # Stop any running preview worker
        if self.preview_worker is not None and self.preview_worker.isRunning():
            self.preview_worker.stop()
            self.preview_worker.wait()

        self._prev_frames.clear()
        self._prev_current_idx = 0
        self._prev_auto_fit_pending = True
        self._frame_slider.setRange(0, 0)
        self._frame_label.setText("0/0")
        self.status_label.setText(f"Generating preview for Rank {row + 1}\u2026")

        self.preview_worker = BgDetectionPreviewWorker(
            video_path=self.video_path,
            base_params=self.base_params,
            trial_params=trial_params,
            n_sample_frames=self.spin_frames.value(),
            parent=self,
            **self._preview_cache_kwargs(),
        )
        self.preview_worker.frame_signal.connect(self._on_preview_frame_received)
        self.preview_worker.finished_signal.connect(
            lambda: self.status_label.setText(
                f"Preview of rank {row + 1} finished "
                f"({len(self._prev_frames)} frames)."
            )
        )
        self.preview_worker.start()

    @Slot(int, object)
    def _on_preview_frame_received(self, idx: int, rgb: np.ndarray) -> None:
        self._prev_frames.append(rgb)
        n = len(self._prev_frames)
        self._prev_current_idx = n - 1
        self._frame_slider.blockSignals(True)
        self._frame_slider.setRange(0, max(n - 1, 0))
        self._frame_slider.setValue(n - 1)
        self._frame_slider.blockSignals(False)
        self._frame_label.setText(f"{n}/{n}")
        self._display_preview_frame(rgb)
        if self._prev_auto_fit_pending:
            self._prev_auto_fit_pending = False
            QTimer.singleShot(0, self._fit_preview)

    def _on_frame_slider_changed(self, val: int) -> None:
        if 0 <= val < len(self._prev_frames):
            self._prev_current_idx = val
            self._frame_label.setText(f"{val + 1}/{len(self._prev_frames)}")
            self._display_preview_frame(self._prev_frames[val])

    def _on_frame_slider_moved(self, val: int) -> None:
        if self._frame_slider.hasTracking():
            return
        if 0 <= val < len(self._prev_frames):
            self._frame_label.setText(f"{val + 1}/{len(self._prev_frames)}")

    # ── Preview display ───────────────────────────────────────────────────────

    def _display_preview_frame(self, rgb: np.ndarray) -> None:
        h, w = rgb.shape[:2]
        z = self._prev_zoom_slider.value() / 100.0
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        transform_mode = (
            Qt.FastTransformation
            if self._frame_slider.isSliderDown()
            or (self.preview_worker is not None and self.preview_worker.isRunning())
            else Qt.SmoothTransformation
        )
        scaled = qimg.scaled(
            int(w * z),
            int(h * z),
            Qt.KeepAspectRatio,
            transform_mode,
        )
        pix = QPixmap.fromImage(scaled)
        self._prev_label.setPixmap(pix)
        self._prev_label.resize(pix.width(), pix.height())

    def _on_prev_zoom_changed(self, val: int) -> None:
        self._prev_zoom_label.setText(f"{val}%")
        if self._prev_frames and 0 <= self._prev_current_idx < len(self._prev_frames):
            self._display_preview_frame(self._prev_frames[self._prev_current_idx])
            self._restore_preview_zoom_anchor()

    def _fit_preview(self) -> None:
        if not self._prev_frames:
            return
        rgb = self._prev_frames[self._prev_current_idx]
        h, w = rgb.shape[:2]
        avail_w = max(self._prev_scroll.width() - 4, 1)
        avail_h = max(self._prev_scroll.height() - 4, 1)
        z = min(avail_w / max(w, 1), avail_h / max(h, 1))
        self._prev_zoom_slider.setValue(max(10, min(400, int(z * 100))))

    def _capture_preview_zoom_anchor(self, viewport_pos=None) -> None:
        viewport = self._prev_scroll.viewport()
        if viewport_pos is None:
            viewport_pos = viewport.rect().center()
        viewport_x = max(0, min(viewport.width(), viewport_pos.x()))
        viewport_y = max(0, min(viewport.height(), viewport_pos.y()))
        label_width = max(self._prev_label.width(), 1)
        label_height = max(self._prev_label.height(), 1)
        offset_x = max((viewport.width() - label_width) // 2, 0)
        offset_y = max((viewport.height() - label_height) // 2, 0)
        local_x = (
            self._prev_scroll.horizontalScrollBar().value() + viewport_x - offset_x
        )
        local_y = self._prev_scroll.verticalScrollBar().value() + viewport_y - offset_y
        local_x = max(0, min(label_width, local_x))
        local_y = max(0, min(label_height, local_y))
        self._prev_zoom_anchor = (
            viewport_x,
            viewport_y,
            local_x / label_width,
            local_y / label_height,
        )

    def _restore_preview_zoom_anchor(self) -> None:
        if self._prev_zoom_anchor is None:
            return
        viewport_x, viewport_y, rel_x, rel_y = self._prev_zoom_anchor
        self._prev_zoom_anchor = None
        viewport = self._prev_scroll.viewport()
        label_width = max(self._prev_label.width(), 1)
        label_height = max(self._prev_label.height(), 1)
        offset_x = max((viewport.width() - label_width) // 2, 0)
        offset_y = max((viewport.height() - label_height) // 2, 0)
        target_x = int(round(rel_x * label_width - viewport_x + offset_x))
        target_y = int(round(rel_y * label_height - viewport_y + offset_y))
        hbar = self._prev_scroll.horizontalScrollBar()
        vbar = self._prev_scroll.verticalScrollBar()
        hbar.setValue(max(0, min(hbar.maximum(), target_x)))
        vbar.setValue(max(0, min(vbar.maximum(), target_y)))

    def _preview_viewport_pos_from_event(self, evt, source_widget=None):
        viewport = self._prev_scroll.viewport()
        origin = source_widget or self._prev_label
        if hasattr(evt, "position"):
            pos = evt.position()
            local_point = pos.toPoint() if hasattr(pos, "toPoint") else pos
            if origin is viewport:
                return local_point
            return viewport.mapFromGlobal(origin.mapToGlobal(local_point))
        if hasattr(evt, "globalPosition"):
            pos = evt.globalPosition()
            global_point = pos.toPoint() if hasattr(pos, "toPoint") else pos
            return viewport.mapFromGlobal(global_point)
        return viewport.rect().center()

    def _set_preview_zoom_value(self, new_value: int, viewport_pos=None) -> bool:
        bounded_value = max(10, min(400, int(round(new_value))))
        if bounded_value == self._prev_zoom_slider.value():
            return False
        if viewport_pos is not None:
            self._capture_preview_zoom_anchor(viewport_pos)
        self._prev_zoom_slider.setValue(bounded_value)
        return True

    def _handle_preview_wheel(self, evt, source_widget=None) -> bool:
        if evt.modifiers() != Qt.ControlModifier:
            return False
        viewport_pos = self._preview_viewport_pos_from_event(evt, source_widget)
        delta = evt.angleDelta().y()
        if delta == 0:
            evt.accept()
            return True
        zoom_step = 10 if delta > 0 else -10
        self._set_preview_zoom_value(
            self._prev_zoom_slider.value() + zoom_step,
            viewport_pos,
        )
        evt.accept()
        return True

    def _handle_preview_native_gesture(self, evt, source_widget=None) -> bool:
        gesture_type = evt.gestureType()
        zoom_gesture = getattr(Qt, "ZoomNativeGesture", None)
        begin_gesture = getattr(Qt, "BeginNativeGesture", None)
        end_gesture = getattr(Qt, "EndNativeGesture", None)
        if gesture_type in (begin_gesture, end_gesture):
            evt.accept()
            return True
        if gesture_type != zoom_gesture:
            return False
        viewport_pos = self._preview_viewport_pos_from_event(evt, source_widget)
        scale_delta = float(evt.value())
        if abs(scale_delta) < 1e-6:
            evt.accept()
            return True
        scaled_zoom = self._prev_zoom_slider.value() * max(0.2, 1.0 + scale_delta)
        if int(round(scaled_zoom)) == self._prev_zoom_slider.value():
            scaled_zoom = self._prev_zoom_slider.value() + (
                1 if scale_delta > 0 else -1
            )
        self._set_preview_zoom_value(scaled_zoom, viewport_pos)
        evt.accept()
        return True

    def _handle_preview_pinch_gesture(self, evt, source_widget=None) -> bool:
        pinch = evt.gesture(Qt.PinchGesture)
        if pinch is None:
            return False
        if pinch.state() == Qt.GestureUpdated:
            viewport_pos = self._preview_viewport_pos_from_event(pinch, source_widget)
            zoom_delta = int((pinch.scaleFactor() - 1.0) * 60)
            if zoom_delta != 0:
                self._set_preview_zoom_value(
                    self._prev_zoom_slider.value() + zoom_delta,
                    viewport_pos,
                )
        evt.accept()
        return True

    def eventFilter(self, watched, event):
        preview_targets = (self._prev_label, self._prev_scroll.viewport())
        if watched in preview_targets:
            if event.type() == QEvent.Wheel and self._handle_preview_wheel(
                event, watched
            ):
                return True
            if (
                event.type() == QEvent.NativeGesture
                and self._handle_preview_native_gesture(event, watched)
            ):
                return True
            if event.type() == QEvent.Gesture and self._handle_preview_pinch_gesture(
                event, watched
            ):
                return True
        return super().eventFilter(watched, event)

    # ── Preview mouse events ──────────────────────────────────────────────────

    def _on_prev_mouse_press(self, evt) -> None:
        if evt.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._prev_panning = True
            self._prev_pan_start = evt.globalPosition().toPoint()
            self._prev_scroll_h = self._prev_scroll.horizontalScrollBar().value()
            self._prev_scroll_v = self._prev_scroll.verticalScrollBar().value()
            self._prev_label.setCursor(Qt.ClosedHandCursor)
            evt.accept()

    def _on_prev_mouse_move(self, evt) -> None:
        if self._prev_panning and self._prev_pan_start is not None:
            delta = evt.globalPosition().toPoint() - self._prev_pan_start
            self._prev_scroll.horizontalScrollBar().setValue(
                self._prev_scroll_h - delta.x()
            )
            self._prev_scroll.verticalScrollBar().setValue(
                self._prev_scroll_v - delta.y()
            )
            evt.accept()
        else:
            self._prev_label.setCursor(Qt.OpenHandCursor)

    def _on_prev_mouse_release(self, evt) -> None:
        if self._prev_panning:
            self._prev_panning = False
            self._prev_pan_start = None
            self._prev_label.setCursor(Qt.OpenHandCursor)
            evt.accept()

    def _on_prev_wheel(self, evt) -> None:
        if not self._handle_preview_wheel(evt, self._prev_label):
            evt.ignore()  # pass to scroll area for normal scrolling

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # noqa: N802
        """Stop any running optimizer or preview worker before closing the dialog."""
        if self.optimizer is not None and self.optimizer.isRunning():
            self.optimizer.stop()
            self.optimizer.wait(3000)
        if self.preview_worker is not None and self.preview_worker.isRunning():
            self.preview_worker.stop()
            self.preview_worker.wait(3000)
        super().closeEvent(event)
