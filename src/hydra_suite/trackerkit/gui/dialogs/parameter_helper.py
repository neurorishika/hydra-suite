"""
Parameter Selection Helper Dialog.
Integrated with Main UI frame range and granular parameter control.
"""

import hashlib
import json
import logging
from pathlib import Path
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

from hydra_suite.core.tracking.optimizer import (
    _PARAM_RANGES,
    OptimizationResult,
    TrackingOptimizer,
)
from hydra_suite.core.tracking.optimizer_workers import TrackingPreviewWorker
from hydra_suite.utils.video_artifacts import build_autotune_state_path
from hydra_suite.widgets.dialogs import BaseDialog

logger = logging.getLogger(__name__)

# ── Colour helpers ────────────────────────────────────────────────────────────


def _score_to_color(value: float) -> QColor:
    """Return green→yellow→red QColor for a [0, 1] sub-score (0=good/green, 1=bad/red)."""
    v = max(0.0, min(1.0, value))
    if v < 0.5:
        r = int(v * 2 * 200)
        g = 160
    else:
        r = 160
        g = int((1.0 - v) * 2 * 160)
    return QColor(r, g, 60, 220)


def _badge_item(cost: float, display: str) -> QTableWidgetItem:
    """Colour-coded table cell; cost in [0,1] where 0=best."""
    item = QTableWidgetItem(display)
    item.setBackground(QBrush(_score_to_color(cost)))
    item.setForeground(QBrush(QColor(255, 255, 255)))
    f = QFont()
    f.setFamily("Monospace")
    f.setPointSize(8)
    item.setFont(f)
    item.setTextAlignment(Qt.AlignCenter)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item


# ── Dialog ────────────────────────────────────────────────────────────────────


class ParameterHelperDialog(BaseDialog):
    """Interactive dialog for auto-tuning core tracking parameters (Kalman, YOLO thresholds, assignment weights) against a detection cache."""

    def __init__(
        self,
        video_path: str,
        detection_cache_path: str,
        start_frame: int,
        end_frame: int,
        current_params: Dict[str, Any],
        parent=None,
    ):
        super().__init__(
            title="Tracking Auto-Tuner",
            parent=parent,
            buttons=QDialogButtonBox.NoButton,
            apply_dark_style=True,
        )
        self.video_path = video_path
        self.detection_cache_path = detection_cache_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.base_params = current_params.copy()

        self.results: List[OptimizationResult] = []
        self.optimizer: TrackingOptimizer | None = None
        self.preview_worker: TrackingPreviewWorker | None = None

        # Preview panel state
        self._prev_panning = False
        self._prev_pan_start = None
        self._prev_scroll_h = 0
        self._prev_scroll_v = 0
        self._prev_last_frame: np.ndarray | None = None
        self._prev_pending_frame: np.ndarray | None = None
        self._prev_render_scheduled = False
        self._prev_zoom_anchor: tuple[int, int, float, float] | None = None
        self._prev_auto_fit_pending = True

        self.setMinimumSize(1200, 740)
        self.setup_ui()
        self._load_state()

    # ── UI construction ───────────────────────────────────────────────────────

    def setup_ui(self):
        """Build the full dialog layout: left controls/results table and right live-preview panel."""
        container = QWidget()
        root = QVBoxLayout(container)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # ── Outer splitter: left (controls + table) | right (preview, full height) ──
        outer = QSplitter(Qt.Horizontal)
        outer.setChildrenCollapsible(False)

        # ── Left panel ────────────────────────────────────────────────────────
        left_w = QWidget()
        left = QVBoxLayout(left_w)
        left.setSpacing(6)
        left.setContentsMargins(6, 6, 4, 6)

        # Header
        hdr = QLabel(
            f"Optimizing range: <b>{self.start_frame} – {self.end_frame}</b>"
            f"  ({self.end_frame - self.start_frame + 1} frames)"
        )
        hdr.setStyleSheet("font-size: 12px; color: #9cdcfe; margin-bottom: 4px;")
        left.addWidget(hdr)

        # ── Domain Constraints ────────────────────────────────────────────────
        # Read-only summary of the physical parameters that are set in the Main
        # Window tracking tab ("Track Continuity" section).  The optimiser uses
        # these values as fixed constraints and never tunes them.  If any look
        # wrong, close this dialog, adjust them in the Main Window, and reopen.
        domain_box = QGroupBox(
            "Physical Constraints  \u2014  read from Main Window (close & adjust there if needed)"
        )
        domain_box.setStyleSheet(
            "QGroupBox { border: 1px solid #7a5f20; border-radius: 4px;"
            " margin-top: 6px; padding-top: 4px; color: #f0c060; font-weight: bold; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
        )
        domain_lay = QHBoxLayout(domain_box)
        domain_lay.setSpacing(0)
        domain_lay.setContentsMargins(8, 4, 8, 4)

        _body_px = self.base_params.get(
            "REFERENCE_BODY_SIZE", 20.0
        ) * self.base_params.get("RESIZE_FACTOR", 1.0)
        _vel_mult = self.base_params.get("KALMAN_MAX_VELOCITY_MULTIPLIER", 2.0)
        _aniso = self.base_params.get("KALMAN_ANISOTROPY_RATIO", 10.0)
        summary = QLabel(
            f"\u25cf\u00a0Body size: <b>{_body_px:.1f}\u00a0px</b>"
            f"\u2002\u2502\u2002"
            f"\u25cf\u00a0Max velocity: <b>{_vel_mult:.1f}\u00d7\u00a0body/frame</b>"
            f"\u2002\u2502\u2002"
            f"\u25cf\u00a0Motion anisotropy\u00a0(fwd\u00f7lat): <b>{_aniso:.1f}</b>"
        )
        summary.setTextFormat(Qt.RichText)
        summary.setStyleSheet("font-size: 11px; color: #ffffff; font-weight: normal;")
        summary.setToolTip(
            "These values come directly from the Main Window tracking tab.\n"
            "Body size         \u2192 REFERENCE_BODY_SIZE \u00d7 RESIZE_FACTOR\n"
            "Max velocity      \u2192 'Max speed' spinbox (Kalman section)\n"
            "Motion anisotropy \u2192 derived from Longitudinal \u00f7 Lateral noise spinboxes\n\n"
            "Close this dialog, change those values, then reopen to use different constraints.\n"
            "Changing them will invalidate any cached autotune results."
        )
        domain_lay.addWidget(summary)
        domain_lay.addStretch()
        left.addWidget(domain_box)

        # Parameter tabs — two tabs: scoring weights first, then all parameter groups
        tab = QTabWidget()
        tab.setStyleSheet("QTabBar::tab { padding: 5px 14px; }")
        tab.addTab(self._make_scoring_tab(), "⚖  Scoring Weights")
        tab.addTab(self._make_params_tab(), "Optimization Parameters")
        left.addWidget(tab)

        # Optimization control bar
        left.addLayout(self._make_opt_bar())

        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        left.addWidget(self.progress)

        self.status_label = QLabel("Select parameters above, then click Run.")
        self.status_label.setStyleSheet("color: #ffffff; font-size: 11px;")
        left.addWidget(self.status_label)

        # Results table — 9 columns: Rank | Score | Cov↑ | Asn↓ | Frg↓ | Occ↓ | Vel↓ | Crd↓ | Key Changes
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            [
                "Rank",
                "Score",
                "Cov↑",
                "Asn↓",
                "Frg↓",
                "Occ↓",
                "Vel↓",
                "Crd↓",
                "Key Parameter Changes",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(8, QHeaderView.Stretch)
        for col in range(8):
            self.table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeToContents
            )
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setDefaultSectionSize(36)
        left.addWidget(self.table, stretch=1)

        # Bottom buttons
        left.addLayout(self._make_bottom_bar())

        outer.addWidget(left_w)

        # ── Right panel — full-height preview ────────────────────────────────
        outer.addWidget(self._make_preview_panel())

        # Give left ~60 % and preview ~40 % of initial width
        outer.setSizes([800, 560])

        root.addWidget(outer)
        self.add_content(container)

    # ── Tab builders ──────────────────────────────────────────────────────────

    def _make_params_tab(self) -> QWidget:
        """Combined scrollable tab that stacks Detection / Assignment / Kalman /
        Lifecycle subsections so the Scoring Weights tab has more breathing room."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setSpacing(8)
        lay.setContentsMargins(4, 4, 4, 4)

        for build in [
            self._make_detection_tab,
            self._make_assignment_tab,
            self._make_kalman_tab,
            self._make_lifecycle_tab,
        ]:
            tab_w = build()
            tab_lay = tab_w.layout()
            while tab_lay.count():
                item = tab_lay.takeAt(0)
                if item.widget():
                    lay.addWidget(item.widget())

        lay.addStretch()
        scroll.setWidget(container)
        return scroll

    def _checkboxes_in_group(self, title: str, items: list) -> QGroupBox:
        """Build a 2-column checkbox grid inside a labelled QGroupBox."""
        box = QGroupBox(title)
        grid = QGridLayout(box)
        grid.setSpacing(6)
        for idx, (cb, tooltip) in enumerate(items):
            if tooltip:
                cb.setToolTip(tooltip)
            grid.addWidget(cb, idx // 2, idx % 2)
        return box

    def _make_detection_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        self.cb_conf = QCheckBox("YOLO Confidence threshold")
        self.cb_conf.setChecked(True)
        self.cb_iou = QCheckBox("YOLO IOU threshold  (NMS suppression)")

        lay.addWidget(
            self._checkboxes_in_group(
                "Detection Filtering",
                [
                    (
                        self.cb_conf,
                        "Post-hoc confidence filter applied to the raw detection cache.\n"
                        "Lower = more detections (including false positives).\n"
                        "Higher = fewer, cleaner detections.",
                    ),
                    (
                        self.cb_iou,
                        "Overlap threshold for NMS applied post-hoc to the raw cache.\n"
                        "Lower = more aggressive NMS (removes overlapping boxes).\n"
                        "Higher = more detections survive.",
                    ),
                ],
            )
        )
        lay.addStretch()
        return w

    def _make_assignment_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        self.cb_dist = QCheckBox("Max movement gate  (× body size)")
        self.cb_dist.setChecked(True)
        self.cb_w_pos = QCheckBox("Weight: Position")
        self.cb_w_ori = QCheckBox("Weight: Orientation")
        self.cb_w_area = QCheckBox("Weight: Area")
        self.cb_w_asp = QCheckBox("Weight: Aspect ratio")

        lay.addWidget(
            self._checkboxes_in_group(
                "Matching Gate & Cost Weights",
                [
                    (
                        self.cb_dist,
                        "Maximum predicted → detection distance in units of body size.\n"
                        "Too small = fragmentation on fast or briefly-occluded animals.\n"
                        "Too large = wrong-animal association across the arena.",
                    ),
                    (
                        self.cb_w_pos,
                        "Cost weight for positional distance in the assignment matrix.\n"
                        "Primary assignment cue; usually the most important weight.",
                    ),
                    (
                        self.cb_w_ori,
                        "Cost weight for orientation difference.\n"
                        "Useful for directional animals (fish, ants) but can confuse round ones.",
                    ),
                    (
                        self.cb_w_area,
                        "Cost weight for area (size) difference between prediction and detection.\n"
                        "Helps distinguish large vs. small individuals; set 0 if all sizes are similar.",
                    ),
                    (
                        self.cb_w_asp,
                        "Cost weight for aspect-ratio difference between prediction and detection.\n"
                        "Useful when animals have distinctive elongation.",
                    ),
                ],
            )
        )
        lay.addStretch()
        return w

    def _make_kalman_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        self.cb_kalman_p = QCheckBox("Process noise σ  (filter smoothness)")
        self.cb_kalman_m = QCheckBox("Measurement noise σ  (detection trust)")
        self.cb_kalman_damp = QCheckBox("Velocity damping  (friction per frame)")
        self.cb_kalman_long_noise = QCheckBox(
            "Longitudinal noise scale  (heading process noise)"
        )
        self.cb_kalman_init_vel = QCheckBox(
            "Initial velocity retention  (bootstrap aggressiveness)"
        )
        self.cb_kalman_maturity = QCheckBox(
            "Track maturity age  (velocity bootstrap frames)"
        )

        lay.addWidget(
            self._checkboxes_in_group(
                "Kalman Filter Parameters",
                [
                    (
                        self.cb_kalman_p,
                        "Base process noise covariance (σ).\n"
                        "Higher = filter trusts motion model less → follows detections more aggressively.\n"
                        "Lower = smoother but may lag behind fast animals.",
                    ),
                    (
                        self.cb_kalman_m,
                        "Measurement noise covariance (σ).\n"
                        "Higher = filter trusts detections less → smoother but less responsive.\n"
                        "Lower = snappy but jerky output.",
                    ),
                    (
                        self.cb_kalman_damp,
                        "Velocity friction per frame (range 0.70–0.999).\n"
                        "High (≈0.99): velocity persists → KF overshoots when animals stop.\n"
                        "Low (≈0.75): velocity decays quickly → less look-ahead, but more agile.",
                    ),
                    (
                        self.cb_kalman_long_noise,
                        "Process noise scale in the heading direction.\n"
                        "Higher = filter allows more forward movement per frame.\n"
                        "Lateral noise is derived automatically as long / anisotropy ratio.\n"
                        "Set the anisotropy ratio in the Physical Constraints panel above.",
                    ),
                    (
                        self.cb_kalman_init_vel,
                        "Fraction of the first observed velocity kept when a track is born.\n"
                        "0.0 = new tracks start stationary (safest; no premature drift).\n"
                        "1.0 = new tracks immediately inherit full detection velocity.\n"
                        "Values ≈ 0.1–0.4 work well for most species.",
                    ),
                    (
                        self.cb_kalman_maturity,
                        "Frames before a new track's velocity estimate is fully trusted.\n"
                        "Higher = slower velocity bootstrap, less initial jitter.\n"
                        "Lower = velocity kicks in immediately (may cause early identity swaps).",
                    ),
                ],
            )
        )
        lay.addStretch()
        return w

    def _make_lifecycle_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        self.cb_lost_thresh = QCheckBox(
            "Lost-track threshold  (frames before ID freed)"
        )

        lay.addWidget(
            self._checkboxes_in_group(
                "Track Lifecycle",
                [
                    (
                        self.cb_lost_thresh,
                        "Consecutive unmatched frames before a track transitions to 'lost'\n"
                        "and its identity slot is freed for re-assignment.\n"
                        "Low (2–5): frequent fragmentation – new IDs assigned after brief occlusions.\n"
                        "High (15–25): tracks persist through long occlusions but may produce ghost tracks.",
                    ),
                ],
            )
        )
        lay.addStretch()
        return w

    def _make_scoring_tab(self) -> QWidget:
        """Tab for controlling the relative weight of each objective sub-score."""
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setSpacing(8)

        note = QLabel(
            "Each weight controls how much that objective influences the composite score.  "
            "Weights are normalised to sum to 1.0 at runtime — use any scale.  "
            "Set a weight to <b>0</b> to disable that term entirely "
            "(it is also excluded from the balance-penalty)."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #ffffff; font-size: 11px;")
        outer.addWidget(note)

        # ── Preset selector ──────────────────────────────────────────────────
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self._combo_presets = QComboBox()
        for name in self._WEIGHT_PRESETS:
            self._combo_presets.addItem(name)
        self._combo_presets.setToolTip(
            "Load a curated weight profile for a common recording scenario.\n"
            "You can fine-tune individual weights after applying a preset."
        )
        self._combo_presets.activated.connect(
            lambda _: self._apply_weight_preset(self._combo_presets.currentText())
        )
        preset_row.addWidget(self._combo_presets, stretch=1)
        btn_apply_preset = QPushButton("Apply")
        btn_apply_preset.setToolTip(
            "Load the selected preset into the weight spinboxes."
        )
        btn_apply_preset.clicked.connect(
            lambda: self._apply_weight_preset(self._combo_presets.currentText())
        )
        preset_row.addWidget(btn_apply_preset)
        outer.addLayout(preset_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3a3a;")
        outer.addWidget(sep)

        # ── Weight spinboxes ─────────────────────────────────────────────────
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
                f"<span style='color:#6ec07a'>↑</span> <span style='color:#ffffff'>{up}</span>"
                f"<br><span style='color:#e0875a'>↓</span> <span style='color:#ffffff'>{down}</span>"
            )
            lbl.setWordWrap(True)
            lbl.setTextFormat(Qt.RichText)
            lbl.setStyleSheet("font-size: 10px;")
            return lbl

        self.spin_w_coverage = _spin(
            0.25,
            "Penalises parameter sets where animals are permanently lost.\n"
            "Higher = the optimiser prioritises keeping all N tracks filled.\n"
            "⚠ Too high forces assignments that stretch across the arena → jumps.",
        )
        self.spin_w_assignment = _spin(
            0.15,
            "Penalises large Kalman innovations (prediction vs. measurement gap).\n"
            "Higher = only tight, predictable matches are accepted.\n"
            "Lower = allows loose assignments on fast / erratic animals.",
        )
        self.spin_w_fragmentation = _spin(
            0.20,
            "Penalises short, broken trajectories.\n"
            "Higher = long unbroken runs rewarded over brief drop-outs.",
        )
        self.spin_w_occlusion = _spin(
            0.10,
            "Penalises transient disappearances (occluded frames).\n"
            "Lower is usually fine; brief occlusion is often unavoidable.",
        )
        self.spin_w_velocity = _spin(
            0.20,
            "Penalises sudden large inter-frame position jumps.\n"
            "This is the primary defence against false swaps and ghost tracks.\n"
            "Increase if you see long-distance jumps that resolve next frame.",
        )
        self.spin_w_crowding = _spin(
            0.10,
            "Penalises pairs of active tracks closer than the reference body size.\n"
            "⚠ Set to 0 for naturally-crowding species (ants, fish schools, herds):\n"
            "   physical proximity is normal biology, not an error, and a constant\n"
            "   crowding signal gives the optimiser no useful gradient.",
        )

        rows = [
            (
                "Coverage",
                self.spin_w_coverage,
                _hint(
                    "Detections are unreliable / animals often missed",
                    "Detector is solid — avoid forcing over-eager assignments",
                ),
            ),
            (
                "Assignment",
                self.spin_w_assignment,
                _hint(
                    "Motion is smooth & predictable (open arena, fish, flies)",
                    "Fast or erratic animals where loose KF matches are OK",
                ),
            ),
            (
                "Fragmentation",
                self.spin_w_fragmentation,
                _hint(
                    "Prioritise long unbroken trajectories above all",
                    "Brief trajectory gaps are acceptable (heavy occlusion scenes)",
                ),
            ),
            (
                "Occlusion",
                self.spin_w_occlusion,
                _hint(
                    "Brief disappearances should be penalised more",
                    "Occlusions are normal (tunnels, burrows, crowded arenas)",
                ),
            ),
            (
                "Velocity",
                self.spin_w_velocity,
                _hint(
                    "Strongly penalise inter-frame jumps — primary anti-swap defence",
                    "Animals can make large rapid movements between frames",
                ),
            ),
            (
                "Crowding",
                self.spin_w_crowding,
                _hint(
                    "Animals should stay spatially separated (isolated individuals)",
                    "Set 0 for naturally crowding species (ants, herds, fish schools)",
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

        # ── Footer: sum display + Renormalize + Reset ─────────────────────────
        ctrl = QHBoxLayout()
        self.lbl_weight_sum = QLabel("Sum: 1.00")
        self.lbl_weight_sum.setStyleSheet("font-size: 11px; color: #9cdcfe;")
        ctrl.addWidget(self.lbl_weight_sum)
        ctrl.addStretch(1)

        btn_norm = QPushButton("Renormalize")
        btn_norm.setToolTip(
            "Scale all weights proportionally so they sum to exactly 1.0.\n"
            "Preserves relative importance of each term."
        )
        btn_norm.clicked.connect(self._renormalize_weights)
        ctrl.addWidget(btn_norm)

        btn_reset = QPushButton("Reset to Defaults")
        btn_reset.setToolTip(
            "Restore the default weight set (0.20 / 0.10 / 0.15 / 0.10 / 0.35 / 0.10)"
        )
        btn_reset.clicked.connect(self._reset_weights)
        ctrl.addWidget(btn_reset)
        outer.addLayout(ctrl)

        outer.addStretch()
        return w

    # Weight presets — (coverage, assignment, fragmentation, occlusion, velocity, crowding)
    _WEIGHT_PRESETS: Dict[str, tuple] = {
        "Balanced (default)": (0.20, 0.10, 0.15, 0.10, 0.35, 0.10),
        "Always visible — open arena": (0.10, 0.20, 0.35, 0.05, 0.25, 0.05),
        "Crowded species — ants / fish schools / herds": (
            0.30,
            0.20,
            0.25,
            0.15,
            0.10,
            0.00,
        ),
        "Frequent occlusion — burrows / tunnels": (0.40, 0.10, 0.15, 0.25, 0.10, 0.00),
        "Fast-moving — prevent jump assignments": (0.15, 0.20, 0.20, 0.10, 0.35, 0.00),
        "Sparse / well-separated individuals": (0.15, 0.20, 0.25, 0.10, 0.20, 0.10),
    }

    def _apply_weight_preset(self, name: str):
        preset = self._WEIGHT_PRESETS.get(name)
        if preset is None:
            return
        for spin, val in zip(
            [
                self.spin_w_coverage,
                self.spin_w_assignment,
                self.spin_w_fragmentation,
                self.spin_w_occlusion,
                self.spin_w_velocity,
                self.spin_w_crowding,
            ],
            preset,
        ):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
        self._update_weight_sum()

    def _renormalize_weights(self):
        spins = [
            self.spin_w_coverage,
            self.spin_w_assignment,
            self.spin_w_fragmentation,
            self.spin_w_occlusion,
            self.spin_w_velocity,
            self.spin_w_crowding,
        ]
        total = sum(s.value() for s in spins)
        if total < 1e-9:
            return  # all-zero — nothing to do
        for s in spins:
            s.blockSignals(True)
            s.setValue(round(s.value() / total, 4))
            s.blockSignals(False)
        self._update_weight_sum()

    def _update_weight_sum(self):
        total = (
            self.spin_w_coverage.value()
            + self.spin_w_assignment.value()
            + self.spin_w_fragmentation.value()
            + self.spin_w_occlusion.value()
            + self.spin_w_velocity.value()
            + self.spin_w_crowding.value()
        )
        color = "#9cdcfe" if abs(total - 1.0) < 0.01 else "#f0a050"
        self.lbl_weight_sum.setText(f"Sum: {total:.2f}")
        self.lbl_weight_sum.setStyleSheet(f"font-size: 11px; color: {color};")

    def _reset_weights(self):
        self._apply_weight_preset("Balanced (default)")

    def get_scoring_weights(self) -> Dict[str, float]:
        """Return the current scoring-weight spinbox values as a params dict."""
        return {
            "SCORE_WEIGHT_COVERAGE": self.spin_w_coverage.value(),
            "SCORE_WEIGHT_ASSIGNMENT": self.spin_w_assignment.value(),
            "SCORE_WEIGHT_FRAGMENTATION": self.spin_w_fragmentation.value(),
            "SCORE_WEIGHT_OCCLUSION": self.spin_w_occlusion.value(),
            "SCORE_WEIGHT_VELOCITY": self.spin_w_velocity.value(),
            "SCORE_WEIGHT_CROWDING": self.spin_w_crowding.value(),
        }

    # ── Control bars ──────────────────────────────────────────────────────────

    def _make_opt_bar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        row.addWidget(QLabel("Trials:"))
        self.spin_trials = QSpinBox()
        self.spin_trials.setRange(10, 2000)
        self.spin_trials.setValue(100)
        self.spin_trials.setToolTip(
            "Total number of Bayesian optimisation trials.\n"
            "50–100 is fast; 200–500 finds better solutions; 1000+ for thorough sweeps."
        )
        row.addWidget(self.spin_trials)

        row.addWidget(QLabel("Seeds:"))
        self.spin_seeds = QSpinBox()
        self.spin_seeds.setRange(1, 10)
        self.spin_seeds.setValue(3)
        self.spin_seeds.setToolTip(
            "Number of diverse starting points queued before optimisation begins.\n"
            "Seed 1 is always your current MAT parameters.\n"
            "Seeds 2–N are random samples spread across the full search space.\n"
            "More seeds = wider initial coverage, less likely to get stuck."
        )
        row.addWidget(self.spin_seeds)

        row.addWidget(QLabel("Plateau:"))
        self.combo_plateau = QComboBox()
        self.combo_plateau.addItems(["Restart (recommended)", "Stop"])
        self.combo_plateau.setToolTip(
            "What to do when no improvement is found for ~20% of the trial budget.\n"
            "Restart: inject a fresh random point and keep searching (finds more).\n"
            "Stop:    terminate early and return results so far (faster)."
        )
        row.addWidget(self.combo_plateau)

        row.addWidget(QLabel("Sampler:"))
        self.combo_sampler = QComboBox()
        self.combo_sampler.addItems(
            [
                "Auto (recommended)",
                "GP — best ≤200 trials",
                "TPE — best >500 trials",
            ]
        )
        self.combo_sampler.setToolTip(
            "Bayesian optimisation algorithm.\n"
            "Auto: OptunaHub AutoSampler — GP early on, TPE later. Best overall.\n"
            "GP:   Gaussian Process (Matérn-2.5, ARD, log-EI). Fastest convergence\n"
            "      for 50–200 trials; requires scipy + torch.\n"
            "TPE:  Multivariate Tree-structured Parzen Estimator. Robust fallback;\n"
            "      no extra dependencies, scales well beyond 500 trials."
        )
        row.addWidget(self.combo_sampler)

        row.addStretch(1)

        self.btn_run = QPushButton("▶  Run Bayesian Optimizer")
        self.btn_run.clicked.connect(self.run_optimization)
        self.btn_run.setStyleSheet(
            "background-color: #0e639c; color: white; font-weight: bold; padding: 8px 16px;"
        )
        row.addWidget(self.btn_run)

        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_optimization)
        self.btn_stop.setStyleSheet(
            "background-color: #c0392b; color: white; font-weight: bold; padding: 8px 12px;"
        )
        row.addWidget(self.btn_stop)

        return row

    # ── Embedded preview panel ────────────────────────────────────────────────

    def _make_preview_panel(self) -> QWidget:
        """Build the right-side preview pane: scroll area + zoom slider + pan."""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 0, 0, 0)
        lay.setSpacing(4)

        title = QLabel("Live Preview")
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
            "No preview yet.\nRun optimizer, select a result row,\nthen click ▶ Preview Selected."
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

        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(6)
        zoom_lbl = QLabel("Zoom:")
        zoom_lbl.setFixedWidth(38)
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

        hint = QLabel("Ctrl+wheel: zoom  •  drag: pan  •  double-click: fit")
        hint.setStyleSheet("color: #ffffff; font-size: 9px;")
        hint.setAlignment(Qt.AlignCenter)
        lay.addWidget(hint)

        return w

    def _on_prev_zoom_changed(self, val: int):
        self._prev_zoom_label.setText(f"{val}%")
        if self._prev_last_frame is not None:
            self._display_preview_frame(self._prev_last_frame)
            self._restore_preview_zoom_anchor()

    def _fit_preview(self):
        if self._prev_last_frame is None:
            return
        h, w = self._prev_last_frame.shape[:2]
        avail_w = max(self._prev_scroll.width() - 4, 1)
        avail_h = max(self._prev_scroll.height() - 4, 1)
        z = min(avail_w / max(w, 1), avail_h / max(h, 1))
        self._prev_zoom_slider.setValue(max(10, min(400, int(z * 100))))

    def _display_preview_frame(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        z = self._prev_zoom_slider.value() / 100.0
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        transform_mode = (
            Qt.FastTransformation
            if self.preview_worker is not None and self.preview_worker.isRunning()
            else Qt.SmoothTransformation
        )
        scaled = qimg.scaled(int(w * z), int(h * z), Qt.KeepAspectRatio, transform_mode)
        pix = QPixmap.fromImage(scaled)
        self._prev_label.setPixmap(pix)
        self._prev_label.resize(pix.width(), pix.height())

    def _schedule_preview_render(self) -> None:
        if self._prev_render_scheduled:
            return
        self._prev_render_scheduled = True
        QTimer.singleShot(0, self._flush_pending_preview_frame)

    def _flush_pending_preview_frame(self) -> None:
        self._prev_render_scheduled = False
        if self._prev_pending_frame is None:
            return
        frame = self._prev_pending_frame
        self._prev_pending_frame = None
        self._display_preview_frame(frame)
        if self._prev_auto_fit_pending:
            self._prev_auto_fit_pending = False
            QTimer.singleShot(0, self._fit_preview)

    @Slot(object)
    def _on_preview_frame_received(self, rgb):
        """Update the embedded preview panel."""
        self._prev_last_frame = rgb
        self._prev_pending_frame = rgb
        self._schedule_preview_render()

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

    def _on_prev_mouse_press(self, evt):
        if evt.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._prev_panning = True
            self._prev_pan_start = evt.globalPosition().toPoint()
            self._prev_scroll_h = self._prev_scroll.horizontalScrollBar().value()
            self._prev_scroll_v = self._prev_scroll.verticalScrollBar().value()
            self._prev_label.setCursor(Qt.ClosedHandCursor)
            evt.accept()

    def _on_prev_mouse_move(self, evt):
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

    def _on_prev_mouse_release(self, evt):
        if self._prev_panning:
            self._prev_panning = False
            self._prev_pan_start = None
            self._prev_label.setCursor(Qt.OpenHandCursor)
            evt.accept()

    def _on_prev_wheel(self, evt):
        if not self._handle_preview_wheel(evt, self._prev_label):
            evt.ignore()  # pass to scroll area for normal scrolling

    # ── Control bars ──────────────────────────────────────────────────────────

    def _make_bottom_bar(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self.btn_preview = QPushButton("▶  Preview Selected")
        self.btn_preview.setEnabled(False)
        self.btn_preview.setToolTip(
            "Run the selected parameter set on the optimised frame range\n"
            "and show the result in the preview panel on the right."
        )
        self.btn_preview.clicked.connect(self.run_preview)
        row.addWidget(self.btn_preview)
        self.btn_apply = QPushButton("✔  Apply Selected")
        self.btn_apply.setEnabled(False)
        self.btn_apply.setToolTip(
            "Write the selected parameter set back into the MAT setup tab and close."
        )
        self.btn_apply.clicked.connect(self._apply_selected)
        self.btn_apply.setStyleSheet(
            "font-weight: bold; background-color: #28a745; color: white; padding: 8px 16px;"
        )
        row.addWidget(self.btn_apply)

        row.addStretch(1)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        row.addWidget(self.btn_cancel)

        return row

    # ── Config gathering ──────────────────────────────────────────────────────

    def get_tuning_config(self) -> Dict[str, bool]:
        """Return a mapping of tracking parameter names to whether each is selected for Optuna optimization."""
        return {
            "YOLO_CONFIDENCE_THRESHOLD": self.cb_conf.isChecked(),
            "YOLO_IOU_THRESHOLD": self.cb_iou.isChecked(),
            "MAX_DISTANCE_MULTIPLIER": self.cb_dist.isChecked(),
            "W_POSITION": self.cb_w_pos.isChecked(),
            "W_ORIENTATION": self.cb_w_ori.isChecked(),
            "W_AREA": self.cb_w_area.isChecked(),
            "W_ASPECT": self.cb_w_asp.isChecked(),
            "KALMAN_NOISE_COVARIANCE": self.cb_kalman_p.isChecked(),
            "KALMAN_MEASUREMENT_NOISE_COVARIANCE": self.cb_kalman_m.isChecked(),
            "KALMAN_DAMPING": self.cb_kalman_damp.isChecked(),
            "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": self.cb_kalman_long_noise.isChecked(),
            "KALMAN_INITIAL_VELOCITY_RETENTION": self.cb_kalman_init_vel.isChecked(),
            "KALMAN_MATURITY_AGE": self.cb_kalman_maturity.isChecked(),
            "LOST_THRESHOLD_FRAMES": self.cb_lost_thresh.isChecked(),
        }

    # ── Optimization ──────────────────────────────────────────────────────────

    def run_optimization(self):
        """Start the Optuna-based tracking optimizer with the selected parameters and trial budget."""
        config = self.get_tuning_config()
        if not any(config.values()):
            QMessageBox.warning(
                self,
                "Selection Required",
                "Please select at least one parameter to tune.",
            )
            return
        if not self.detection_cache_path:
            QMessageBox.critical(
                self,
                "No Cache",
                "Optimization requires a detection cache. Run tracking first.",
            )
            return

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_preview.setEnabled(False)
        self.btn_apply.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.table.setRowCount(0)

        # Merge current scoring weights into base_params so the optimizer's
        # _run_tracking_loop picks them up via params.get("SCORE_WEIGHT_*").
        opt_params = self.base_params.copy()
        opt_params.update(self.get_scoring_weights())

        self.optimizer = TrackingOptimizer(
            self.video_path,
            self.detection_cache_path,
            self.start_frame,
            self.end_frame,
            opt_params,
            config,
            n_trials=self.spin_trials.value(),
            n_seeds=self.spin_seeds.value(),
            on_plateau="stop" if self.combo_plateau.currentIndex() == 1 else "restart",
            sampler_type=["auto", "gp", "tpe"][self.combo_sampler.currentIndex()],
        )
        self.optimizer.progress_signal.connect(self.on_progress)
        self.optimizer.result_signal.connect(self.on_results)
        self.optimizer.finished_signal.connect(self.on_finished)
        self.optimizer.start()

    def _stop_optimization(self):
        if self.optimizer and self.optimizer.isRunning():
            self.optimizer.stop()
        self.btn_stop.setEnabled(False)

    @Slot(int, str)
    def on_progress(self, val: int, msg: str):
        """Update the progress bar and status label with the optimizer's current trial progress."""
        self.progress.setValue(val)
        self.status_label.setText(msg)

    @Slot(list)
    def on_results(self, results: List[OptimizationResult]):
        """Populate the results table with ranked optimization trials and colour-coded sub-score badges."""
        self.results = results
        n_show = min(len(results), 50)
        self.table.setRowCount(n_show)

        for i, res in enumerate(results[:n_show]):
            # Rank
            rank_item = QTableWidgetItem(str(i + 1))
            rank_item.setTextAlignment(Qt.AlignCenter)
            rank_item.setFlags(rank_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 0, rank_item)

            # Composite score (colour-coded, lower=better)
            score_item = QTableWidgetItem(f"{res.score:.4f}")
            score_item.setTextAlignment(Qt.AlignCenter)
            score_item.setBackground(QBrush(_score_to_color(min(res.score / 0.8, 1.0))))
            score_item.setForeground(QBrush(QColor(255, 255, 255)))
            score_item.setFlags(score_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 1, score_item)

            # Sub-score badges  (columns 2-7)
            ss = res.sub_scores
            cov_cost = ss.get("coverage", 1.0)
            cov_pct = 1.0 - cov_cost  # higher = better for display
            self.table.setItem(i, 2, _badge_item(cov_cost, f"{cov_pct:.0%}"))
            self.table.setItem(
                i,
                3,
                _badge_item(
                    ss.get("assignment", 1.0), f"{ss.get('assignment', 1.0):.2f}"
                ),
            )
            self.table.setItem(
                i,
                4,
                _badge_item(
                    ss.get("fragmentation", 1.0), f"{ss.get('fragmentation', 1.0):.2f}"
                ),
            )
            self.table.setItem(
                i,
                5,
                _badge_item(
                    ss.get("occlusion", 1.0), f"{ss.get('occlusion', 1.0):.2f}"
                ),
            )
            self.table.setItem(
                i,
                6,
                _badge_item(ss.get("velocity", 1.0), f"{ss.get('velocity', 1.0):.2f}"),
            )
            self.table.setItem(
                i,
                7,
                _badge_item(ss.get("crowding", 0.0), f"{ss.get('crowding', 0.0):.2f}"),
            )

            # Key parameter changes vs base
            changes = self._format_changes(res.params)
            chg_item = QTableWidgetItem(changes)
            chg_item.setFlags(chg_item.flags() & ~Qt.ItemIsEditable)
            chg_item.setToolTip(changes.replace("  ", "\n"))
            self.table.setItem(i, 8, chg_item)

        if results:
            self.btn_apply.setEnabled(True)
            self.btn_preview.setEnabled(True)
            self.table.selectRow(0)
            self._save_state()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_changes(self, params: Dict[str, Any]) -> str:
        """Compact one-liner: tuned value with ▲/▼/= indicator vs base params."""
        labels = {
            "YOLO_CONFIDENCE_THRESHOLD": ("Conf", ".2f"),
            "YOLO_IOU_THRESHOLD": ("IOU", ".2f"),
            "MAX_DISTANCE_MULTIPLIER": ("Dist", ".1f"),
            "KALMAN_NOISE_COVARIANCE": ("ProcN", ".4f"),
            "KALMAN_MEASUREMENT_NOISE_COVARIANCE": ("MeasN", ".4f"),
            "W_POSITION": ("Wp", ".2f"),
            "W_ORIENTATION": ("Wo", ".2f"),
            "W_AREA": ("Wa", ".2f"),
            "W_ASPECT": ("Wasp", ".2f"),
            "KALMAN_DAMPING": ("Damp", ".3f"),
            "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": ("LongQ", ".1f"),
            "KALMAN_INITIAL_VELOCITY_RETENTION": ("InitV", ".2f"),
            "LOST_THRESHOLD_FRAMES": ("LostF", "d"),
            "KALMAN_MATURITY_AGE": ("Mat", "d"),
        }
        parts = []
        for key, (abbr, fmt) in labels.items():
            if key not in params:
                continue
            val = params[key]
            base_val = self.base_params.get(key)
            val_str = f"{val:{fmt}}"
            if base_val is not None:
                try:
                    diff = float(val) - float(base_val)
                    arrow = "▲" if diff > 1e-9 else ("▼" if diff < -1e-9 else "=")
                    parts.append(f"{abbr}:{val_str}{arrow}")
                except (TypeError, ValueError):
                    parts.append(f"{abbr}:{val_str}")
            else:
                parts.append(f"{abbr}:{val_str}")
        return "  ".join(parts)

    def on_finished(self):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setVisible(False)
        n = len(self.results)
        if n > 0:
            converged = self.optimizer is not None and self.optimizer._stop_requested
            reason = "Converged (plateau)" if converged else "Search finished"
            self.status_label.setText(f"{reason}. {n} trials. Lower score is better.")
        else:
            self.status_label.setText("Search finished with no results.")

    def run_preview(self):
        row = self.table.currentRow()
        if row < 0:
            return
        res = self.results[row]
        preview_params = self.base_params.copy()
        preview_params.update(res.params)

        # Scale dist
        if "MAX_DISTANCE_MULTIPLIER" in res.params:
            ref = preview_params.get("REFERENCE_BODY_SIZE", 20.0)
            rf = preview_params.get("RESIZE_FACTOR", 1.0)
            preview_params["MAX_DISTANCE_THRESHOLD"] = (
                res.params["MAX_DISTANCE_MULTIPLIER"] * ref * rf
            )

        if self.preview_worker and self.preview_worker.isRunning():
            self.preview_worker.stop()
            self.preview_worker.wait()

        self.status_label.setText(f"Previewing Rank {row + 1}...")
        self._prev_auto_fit_pending = True
        self.preview_worker = TrackingPreviewWorker(
            self.video_path,
            self.detection_cache_path,
            self.start_frame,
            self.end_frame,
            preview_params,
        )
        self.preview_worker.frame_signal.connect(self._on_preview_frame_received)
        self.preview_worker.finished_signal.connect(
            lambda: self.status_label.setText(f"Preview of rank {row + 1} finished.")
        )
        self.preview_worker.start()

    def _apply_selected(self):
        """Apply the currently-selected table row (not always rank 1) to MAT."""
        row = self.table.currentRow()
        if row < 0:
            row = 0  # fallback to best
        self._selected_row_to_apply = row
        self.accept()

    def get_selected_params(self) -> Dict[str, Any]:
        row = getattr(self, "_selected_row_to_apply", self.table.currentRow())
        if 0 <= row < len(self.results):
            # Return the full merged set (base snapshot + tuned overrides) so the
            # caller gets exactly the params that produced this result, not just
            # the tuned subset.  The base snapshot also has correct non-tunable
            # values (REFERENCE_BODY_SIZE, RESIZE_FACTOR, etc.) from the time the
            # dialog was opened, preventing any precision or staleness issues.
            return {**self.base_params, **self.results[row].params}
        return {}

    # ── Persistence ───────────────────────────────────────────────────────────

    def _state_path(self) -> Path:
        """Sidecar file next to the detection cache."""
        return build_autotune_state_path(self.detection_cache_path)

    def _compute_state_key(self) -> str:
        """SHA-256 of base_params (tunable keys + domain constraints) + frame range.

        Domain params are physical constraints read from the Main Window that the
        optimiser never tunes but that change the meaning of the score: body size
        (via REFERENCE_BODY_SIZE × RESIZE_FACTOR), max velocity, recovery distance,
        and motion anisotropy.  Changing any of them in the Main Window will
        invalidate cached results when the dialog is reopened.
        """
        subset = {
            k: self.base_params[k]
            for k in sorted(_PARAM_RANGES.keys())
            if k in self.base_params
        }
        domain = {
            "KALMAN_MAX_VELOCITY_MULTIPLIER": self.base_params.get(
                "KALMAN_MAX_VELOCITY_MULTIPLIER", 2.0
            ),
            "KALMAN_ANISOTROPY_RATIO": self.base_params.get(
                "KALMAN_ANISOTROPY_RATIO", 10.0
            ),
            "REFERENCE_BODY_SIZE": self.base_params.get("REFERENCE_BODY_SIZE", 20.0),
            "RESIZE_FACTOR": self.base_params.get("RESIZE_FACTOR", 1.0),
        }
        payload = json.dumps(
            {
                "cache_path": str(self.detection_cache_path),
                "start": self.start_frame,
                "end": self.end_frame,
                "base_params": subset,
                "domain_params": domain,
            },
            sort_keys=True,
            default=str,
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    # Checkbox param-name → widget-attribute mapping
    _CB_MAP = {
        "YOLO_CONFIDENCE_THRESHOLD": "cb_conf",
        "YOLO_IOU_THRESHOLD": "cb_iou",
        "MAX_DISTANCE_MULTIPLIER": "cb_dist",
        "W_POSITION": "cb_w_pos",
        "W_ORIENTATION": "cb_w_ori",
        "W_AREA": "cb_w_area",
        "W_ASPECT": "cb_w_asp",
        "KALMAN_NOISE_COVARIANCE": "cb_kalman_p",
        "KALMAN_MEASUREMENT_NOISE_COVARIANCE": "cb_kalman_m",
        "KALMAN_DAMPING": "cb_kalman_damp",
        "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": "cb_kalman_long_noise",
        "KALMAN_INITIAL_VELOCITY_RETENTION": "cb_kalman_init_vel",
        "KALMAN_MATURITY_AGE": "cb_kalman_maturity",
        "LOST_THRESHOLD_FRAMES": "cb_lost_thresh",
    }

    # Scoring weight key → widget-attribute mapping
    _SW_MAP = {
        "SCORE_WEIGHT_COVERAGE": "spin_w_coverage",
        "SCORE_WEIGHT_ASSIGNMENT": "spin_w_assignment",
        "SCORE_WEIGHT_FRAGMENTATION": "spin_w_fragmentation",
        "SCORE_WEIGHT_OCCLUSION": "spin_w_occlusion",
        "SCORE_WEIGHT_VELOCITY": "spin_w_velocity",
        "SCORE_WEIGHT_CROWDING": "spin_w_crowding",
    }

    def _save_state(self):
        """Write tuning settings + results to the sidecar JSON file."""
        try:
            tuning_cfg = self.get_tuning_config()
            opt_settings = {
                "n_trials": self.spin_trials.value(),
                "n_seeds": self.spin_seeds.value(),
                "plateau": self.combo_plateau.currentIndex(),
                "sampler": self.combo_sampler.currentIndex(),
            }
            scoring_weights = self.get_scoring_weights()
            results_list = [
                {
                    "params": res.params,
                    "score": res.score,
                    "trial_number": res.trial_number,
                    "sub_scores": res.sub_scores,
                }
                for res in self.results
            ]
            state = {
                "cache_key": self._compute_state_key(),
                "tuning_config": tuning_cfg,
                "opt_settings": opt_settings,
                "scoring_weights": scoring_weights,
                "results": results_list,
            }
            path = self._state_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state, indent=2, default=str))
        except Exception:
            logger.exception("Failed to save autotune state")

    def _load_state(self):
        """Restore tuning settings (always) and results (only when key matches)."""
        path = self._state_path()
        if not path.exists():
            return
        try:
            state = json.loads(path.read_text())
        except Exception:
            logger.warning("Could not read autotune state file %s", path)
            return

        # ── Restore UI settings (checkboxes, opt bar, scoring weights) ─────────
        tuning_cfg = state.get("tuning_config", {})
        for param_key, attr in self._CB_MAP.items():
            widget = getattr(self, attr, None)
            if widget is not None and param_key in tuning_cfg:
                widget.setChecked(bool(tuning_cfg[param_key]))

        opt = state.get("opt_settings", {})
        if "n_trials" in opt:
            self.spin_trials.setValue(int(opt["n_trials"]))
        if "n_seeds" in opt:
            self.spin_seeds.setValue(int(opt["n_seeds"]))
        if "plateau" in opt:
            idx = int(opt["plateau"])
            if 0 <= idx < self.combo_plateau.count():
                self.combo_plateau.setCurrentIndex(idx)
        if "sampler" in opt:
            idx = int(opt["sampler"])
            if 0 <= idx < self.combo_sampler.count():
                self.combo_sampler.setCurrentIndex(idx)

        sw = state.get("scoring_weights", {})
        for sw_key, attr in self._SW_MAP.items():
            widget = getattr(self, attr, None)
            if widget is not None and sw_key in sw:
                widget.setValue(float(sw[sw_key]))

        # ── Restore results only if the cache key still matches ───────────────
        saved_key = state.get("cache_key", "")
        current_key = self._compute_state_key()
        if saved_key != current_key:
            self.status_label.setText(
                "\u26a0 Settings restored. Previous results discarded (parameters changed)."
            )
            return

        raw_results = state.get("results", [])
        if not raw_results:
            return

        restored: List[OptimizationResult] = []
        for d in raw_results:
            r = OptimizationResult(
                params=d.get("params", {}),
                score=float(d.get("score", 1.0)),
                trial_number=int(d.get("trial_number", 0)),
                sub_scores=d.get("sub_scores", {}),
            )
            restored.append(r)

        self.on_results(restored)
        self.status_label.setText(
            f"Restored {len(restored)} cached results from previous run."
        )

    def closeEvent(self, event) -> None:
        self._save_state()
        if self.optimizer and self.optimizer.isRunning():
            self.optimizer.stop()
            self.optimizer.wait()
        if self.preview_worker and self.preview_worker.isRunning():
            self.preview_worker.stop()
            self.preview_worker.wait()
        super().closeEvent(event)
