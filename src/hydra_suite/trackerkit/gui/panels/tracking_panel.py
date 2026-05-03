"""TrackingPanel — core tracking parameters, Kalman filter, and assignment config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.trackerkit.config.schemas import TrackerConfig

if TYPE_CHECKING:
    from hydra_suite.trackerkit.gui.main_window import MainWindow


# ─── Hardcoded defaults for parameters formerly exposed in the GUI ──────────
# These were never tuned by the parameter optimizer or in practice. They are
# baked in so the panel stays compact; saved-config keys still flow through
# the runtime params dict so external tests/configs continue to work.

# Kalman lateral noise is auto-derived from longitudinal × anisotropy ratio.
# Ratio matches the previous default (5.0 / 0.1 = 50) and is treated as a
# biological constant for body-axis-aligned motion.
KALMAN_ANISOTROPY_RATIO_CONST = 50.0

# Pose-rejection knobs: never appeared in the optimizer; defaults are universal.
POSE_REJECTION_THRESHOLD_CONST = 0.5
POSE_REJECTION_MIN_VISIBILITY_CONST = 0.5

# Density-map low-level knobs: scale-/dimension-independent defaults.
DENSITY_GAUSSIAN_SIGMA_SCALE_CONST = 1.0
DENSITY_BINARIZE_THRESHOLD_CONST = 0.3
DENSITY_DOWNSAMPLE_FACTOR_CONST = 8

# Solver auto-pick threshold: above this many tracked animals, switch to
# greedy + spatial indexing for speed; below it Hungarian is strictly better.
SOLVER_AUTOPICK_GREEDY_THRESHOLD = 50

# Min-detections-to-start: previously an FPS-coupled spinbox that was
# inadvertently used as a per-frame detection-count threshold. 1 restores
# the documented behaviour ("any frame with ≥1 detection counts toward init").
MIN_DETECTIONS_TO_START_CONST = 1


class TrackingPanel(QWidget):
    """Kalman filter parameters, identity assignment, and backward pass controls."""

    config_changed: Signal = Signal(object)

    def __init__(
        self,
        main_window: "MainWindow",
        config: TrackerConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._main_window = main_window
        self._config = config

        # Hidden/expert parameters: no UI control, but values still flow
        # through saved configs so existing presets keep working unchanged.
        # Defaults match the previous GUI defaults; the orchestrator
        # overwrites these from disk during config load.
        self._kalman_lateral_noise_multiplier = (
            self._kalman_longitudinal_default_for_init() / KALMAN_ANISOTROPY_RATIO_CONST
        )
        self._pose_rejection_threshold = POSE_REJECTION_THRESHOLD_CONST
        self._pose_rejection_min_visibility = POSE_REJECTION_MIN_VISIBILITY_CONST
        self._density_gaussian_sigma_scale = DENSITY_GAUSSIAN_SIGMA_SCALE_CONST
        self._density_binarize_threshold = DENSITY_BINARIZE_THRESHOLD_CONST
        self._density_downsample_factor = DENSITY_DOWNSAMPLE_FACTOR_CONST
        self._min_detections_to_start_seconds = 0.03
        # Solver overrides: None = auto-pick from animal count; bool = saved override.
        self._enable_greedy_override: bool | None = None
        self._enable_spatial_override: bool | None = None

        self._layout = QVBoxLayout(self)
        self._build_ui()

    @staticmethod
    def _kalman_longitudinal_default_for_init() -> float:
        """Default longitudinal-noise value used to seed lateral noise."""
        # Matches spin_kalman_longitudinal_noise default in _build_ui (5.0).
        return 5.0

    def _build_ui(self) -> None:
        from hydra_suite.trackerkit.gui.widgets.collapsible import (
            AccordionContainer,
            CollapsibleGroupBox,
        )

        layout = self._layout
        layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content = QWidget()
        vbox = QVBoxLayout(content)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(8)
        self._main_window._set_compact_scroll_layout(vbox)
        vbox.setAlignment(Qt.AlignTop)

        # ── Basic settings ────────────────────────────────────────────────
        # Always visible at the top: the few knobs most users actually touch.
        # Everything else lives in the collapsed Advanced accordion below.
        g_core = QGroupBox("Basic settings")
        self._main_window._set_compact_section_widget(g_core)
        vl_core = QVBoxLayout(g_core)
        vl_core.setSpacing(8)
        vl_core.addWidget(
            self._main_window._create_help_label(
                "Max movement sets how far an animal can move between frames. "
                "Max speed gates Kalman predictions to physically plausible values. "
                "Reverse pass and the low-confidence map further improve accuracy."
            )
        )
        f_core = QFormLayout(None)
        self._configure_form_layout(f_core)

        self.spin_max_dist = QDoubleSpinBox()
        self.spin_max_dist.setRange(0.1, 20.0)
        self.spin_max_dist.setSingleStep(0.1)
        self.spin_max_dist.setDecimals(2)
        self.spin_max_dist.setValue(1.5)
        self.spin_max_dist.setToolTip(
            "Maximum distance for track-to-detection assignment (×body size).\n"
            "Animals can move at most this distance between frames.\n"
            "Too low = tracks break frequently, Too high = identity swaps.\n"
            "Recommended: 1-2× for normal motion, 3-5× for fast motion."
        )
        self.spin_kalman_max_velocity = QDoubleSpinBox()
        self.spin_kalman_max_velocity.setRange(0.5, 10.0)
        self.spin_kalman_max_velocity.setSingleStep(0.1)
        self.spin_kalman_max_velocity.setDecimals(1)
        self.spin_kalman_max_velocity.setValue(2.0)
        self.spin_kalman_max_velocity.setToolTip(
            "Maximum speed constraint (× body size per frame).\n"
            "Limits how fast any Kalman prediction can move.\n"
            "velocity_max = this_value × reference_body_size (pixels/frame)\n"
            "Lower = more conservative, Higher = allows faster movement.\n"
            "Recommended: 1.5-3.0 depending on animal speed"
        )
        f_core.addRow(
            self._build_field_grid(
                [
                    ("Max movement (body lengths)", self.spin_max_dist),
                    (
                        "Max speed (body lengths/frame)",
                        self.spin_kalman_max_velocity,
                    ),
                ]
            )
        )

        self.chk_enable_backward = QCheckBox("Run reverse pass for better accuracy")
        self.chk_enable_backward.setChecked(True)
        self.chk_enable_backward.setToolTip(
            "Run tracking in reverse (using cached detections) after forward pass to improve accuracy.\n"
            "Forward detections are cached (~10MB/10k frames), then tracking runs backward.\n"
            "No video reversal needed - RAM efficient and faster.\n"
            "Recommended: Enable for best results (takes ~2× time).\n"
            "Disable for faster processing if accuracy is sufficient."
        )
        self.chk_enable_confidence_density_map = QCheckBox(
            "Enable low-confidence detection map"
        )
        self.chk_enable_confidence_density_map.setChecked(True)
        self.chk_enable_confidence_density_map.setToolTip(
            "Build and apply the low-confidence density map during tracking.\n"
            "When enabled, the advanced density-map controls below are shown\n"
            "and density-aware conservative matching is applied.\n"
            "Disable to skip the extra density-map pass entirely."
        )
        f_core.addRow(
            self._build_checkbox_grid(
                [
                    self.chk_enable_backward,
                    self.chk_enable_confidence_density_map,
                ],
                columns=1,
            )
        )
        vl_core.addLayout(f_core)
        vbox.addWidget(g_core)

        # Parameter Helper Button
        self.btn_param_helper = QPushButton("Auto-Tune Tracking Parameters...")
        self.btn_param_helper.clicked.connect(self._main_window._open_parameter_helper)
        self.btn_param_helper.setStyleSheet(
            "background-color: #0e639c; color: white; font-weight: bold; padding: 5px; margin-top: 5px;"
        )
        self.btn_param_helper.setToolTip(
            "Run automated bayesian search to find optimal tracking parameters for your video."
        )
        vbox.addWidget(self.btn_param_helper)

        # ── Advanced settings ─────────────────────────────────────────────
        # All sections below collapse by default. New users should rely on
        # the basic settings + Auto-Tune; expand a section only to override.
        adv_header = QLabel("Advanced settings")
        adv_header.setStyleSheet(
            "font-weight: 700; font-size: 12px; color: #9cdcfe; "
            "margin-top: 10px; margin-bottom: 2px;"
        )
        adv_header.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        vbox.addWidget(adv_header)
        vbox.addWidget(
            self._make_inline_note(
                "Click a section to expand it. Defaults are tuned for most videos, so start with the basic settings and Auto-Tune before overriding advanced controls manually."
            )
        )

        self.tracking_accordion = AccordionContainer()

        # Kalman
        g_kf = CollapsibleGroupBox("How should motion prediction behave?")
        self.tracking_accordion.addCollapsible(g_kf)
        vl_kf = QVBoxLayout()
        vl_kf.addWidget(
            self._main_window._create_help_label(
                "Kalman filter predicts animal positions using motion history. Process noise controls smoothing, "
                "measurement noise controls responsiveness. Age-dependent damping helps stabilize newly initialized tracks."
            )
        )
        f_kf = QFormLayout(None)
        self._configure_form_layout(f_kf)

        self.spin_kalman_noise = QDoubleSpinBox()
        self.spin_kalman_noise.setRange(0.0, 1.0)
        self.spin_kalman_noise.setDecimals(4)
        self.spin_kalman_noise.setSingleStep(0.001)
        self.spin_kalman_noise.setValue(0.03)
        self.spin_kalman_noise.setToolTip(
            "Process noise covariance (0.0-1.0) for motion prediction.\n"
            "Lower = trust motion model more (smooth, may lag).\n"
            "Higher = trust measurements more (responsive, less smooth).\n"
            "Recommended: 0.01-0.05"
        )

        self.spin_kalman_meas = QDoubleSpinBox()
        self.spin_kalman_meas.setRange(0.0, 1.0)
        self.spin_kalman_meas.setDecimals(4)
        self.spin_kalman_meas.setSingleStep(0.001)
        self.spin_kalman_meas.setValue(0.1)
        self.spin_kalman_meas.setToolTip(
            "Measurement noise covariance (0.0-1.0).\n"
            "Lower = trust detections more (accurate, may be jittery).\n"
            "Higher = trust predictions more (smooth, may drift).\n"
            "Recommended: 0.05-0.15"
        )

        f_kf.addRow(
            self._build_field_grid(
                [
                    ("Process noise", self.spin_kalman_noise),
                    ("Measurement noise", self.spin_kalman_meas),
                ]
            )
        )

        self.spin_kalman_damping = QDoubleSpinBox()
        self.spin_kalman_damping.setRange(0.5, 0.999)
        self.spin_kalman_damping.setSingleStep(0.01)
        self.spin_kalman_damping.setDecimals(3)
        self.spin_kalman_damping.setValue(0.95)
        self.spin_kalman_damping.setToolTip(
            "Velocity damping coefficient (0.5-0.99).\n"
            "Lower = faster decay (stop-and-go).\n"
            "Higher = slower decay (continuous motion).\n"
            "Recommended: 0.90-0.95"
        )
        f_kf.addRow("Velocity damping", self.spin_kalman_damping)

        # Age-dependent velocity damping (compact pair: maturity time + retention)
        self.spin_kalman_maturity_age = QDoubleSpinBox()
        self.spin_kalman_maturity_age.setRange(0.01, 2.0)
        self.spin_kalman_maturity_age.setSingleStep(0.02)
        self.spin_kalman_maturity_age.setDecimals(2)
        self.spin_kalman_maturity_age.setValue(0.17)
        self.spin_kalman_maturity_age.setToolTip(
            "Time for a track to reach maturity (seconds).\n"
            "Young tracks use conservative velocity estimates;\n"
            "after this time, tracks use full dynamics.\n"
            "Recommended: 0.10-0.35 s"
        )

        self.spin_kalman_initial_velocity_retention = QDoubleSpinBox()
        self.spin_kalman_initial_velocity_retention.setRange(0.0, 1.0)
        self.spin_kalman_initial_velocity_retention.setSingleStep(0.05)
        self.spin_kalman_initial_velocity_retention.setDecimals(2)
        self.spin_kalman_initial_velocity_retention.setValue(0.2)
        self.spin_kalman_initial_velocity_retention.setToolTip(
            "Initial velocity retention for brand-new tracks (0.0-1.0).\n"
            "0.0 = assume stationary, 1.0 = full velocity.\n"
            "Recommended: 0.1-0.3"
        )

        f_kf.addRow(
            "New-track caution",
            self._build_field_grid(
                [
                    ("Maturity time (s)", self.spin_kalman_maturity_age),
                    (
                        "Initial velocity retention",
                        self.spin_kalman_initial_velocity_retention,
                    ),
                ]
            ),
        )

        self.spin_kalman_longitudinal_noise = QDoubleSpinBox()
        self.spin_kalman_longitudinal_noise.setRange(0.1, 20.0)
        self.spin_kalman_longitudinal_noise.setSingleStep(0.5)
        self.spin_kalman_longitudinal_noise.setDecimals(1)
        self.spin_kalman_longitudinal_noise.setValue(5.0)
        self.spin_kalman_longitudinal_noise.setToolTip(
            "Forward/longitudinal noise multiplier (0.1-20.0).\n"
            "Controls uncertainty along the movement direction.\n"
            f"Lateral uncertainty is locked at 1/{int(KALMAN_ANISOTROPY_RATIO_CONST)} "
            "of this value (body-axis anisotropy is biologically fixed).\n"
            "Recommended: 3.0-7.0"
        )
        f_kf.addRow(
            "Forward/sideways uncertainty (forward scale)",
            self.spin_kalman_longitudinal_noise,
        )

        vl_kf.addLayout(f_kf)
        g_kf.setContentLayout(vl_kf)
        vbox.addWidget(g_kf)
        self._main_window._remember_collapsible_state(
            "tracking.motion_prediction", g_kf
        )

        # Matching cost
        g_weights = CollapsibleGroupBox("How should match scoring work?")
        self.tracking_accordion.addCollapsible(g_weights)
        l_weights = QVBoxLayout()
        l_weights.addWidget(
            self._main_window._create_help_label(
                "This is the core assignment cost used after motion gating. Position does most of the work; "
                "orientation and coarse box geometry help break ties. The track feature settings control how "
                "per-track appearance summaries adapt over time."
            )
        )

        self.spin_Wp = QDoubleSpinBox()
        self.spin_Wp.setRange(0.0, 10.0)
        self.spin_Wp.setValue(1.0)
        self.spin_Wp.setToolTip(
            "Weight for position distance in the assignment cost.\n"
            "Keep this as the dominant term."
        )
        self.spin_Wo = QDoubleSpinBox()
        self.spin_Wo.setRange(0.0, 10.0)
        self.spin_Wo.setValue(1.0)
        self.spin_Wo.setToolTip(
            "Weight for orientation difference in the assignment cost."
        )
        self.spin_Wa = QDoubleSpinBox()
        self.spin_Wa.setRange(0.0, 1.0)
        self.spin_Wa.setSingleStep(0.001)
        self.spin_Wa.setDecimals(4)
        self.spin_Wa.setValue(0.001)
        self.spin_Wa.setToolTip("Weight for area difference in the assignment cost.")
        self.spin_Wasp = QDoubleSpinBox()
        self.spin_Wasp.setRange(0.0, 10.0)
        self.spin_Wasp.setValue(0.1)
        self.spin_Wasp.setToolTip(
            "Weight for aspect-ratio difference in the assignment cost."
        )
        l_weights.addWidget(
            self._build_field_grid(
                [
                    ("Position weight", self.spin_Wp),
                    ("Direction weight", self.spin_Wo),
                    ("Area weight", self.spin_Wa),
                    ("Aspect weight", self.spin_Wasp),
                ]
            )
        )

        self.chk_use_mahal = QCheckBox("Use motion-aware distance (Mahalanobis)")
        self.chk_use_mahal.setChecked(True)
        self.chk_use_mahal.setToolTip(
            "Use Mahalanobis distance instead of Euclidean for the position term.\n"
            "Respects predicted velocity and uncertainty."
        )
        l_weights.addWidget(self.chk_use_mahal)

        self.spin_track_feature_ema_alpha = QDoubleSpinBox()
        self.spin_track_feature_ema_alpha.setRange(0.0, 0.99)
        self.spin_track_feature_ema_alpha.setDecimals(2)
        self.spin_track_feature_ema_alpha.setSingleStep(0.01)
        self.spin_track_feature_ema_alpha.setValue(0.85)
        self.spin_track_feature_ema_alpha.setToolTip(
            "EMA retention for per-track pose prototype.\n"
            "Higher = slower adaptation. Recommended: 0.80-0.95"
        )
        self.spin_assoc_high_conf_threshold = QDoubleSpinBox()
        self.spin_assoc_high_conf_threshold.setRange(0.0, 1.0)
        self.spin_assoc_high_conf_threshold.setDecimals(2)
        self.spin_assoc_high_conf_threshold.setSingleStep(0.05)
        self.spin_assoc_high_conf_threshold.setValue(0.7)
        self.spin_assoc_high_conf_threshold.setToolTip(
            "Minimum detection confidence to update per-track step-size summary.\n"
            "Recommended: 0.6-0.8"
        )
        l_weights.addWidget(
            self._build_field_grid(
                [
                    ("Track EMA α", self.spin_track_feature_ema_alpha),
                    ("High-confidence threshold", self.spin_assoc_high_conf_threshold),
                ]
            )
        )
        g_weights.setContentLayout(l_weights)
        vbox.addWidget(g_weights)
        self._main_window._remember_collapsible_state(
            "tracking.match_scoring", g_weights
        )

        # Candidate gating and pose safeguards
        g_assign = CollapsibleGroupBox("How should candidate matches be filtered?")
        self.tracking_accordion.addCollapsible(g_assign)
        vl_assign = QVBoxLayout()
        vl_assign.addWidget(
            self._main_window._create_help_label(
                "First, the tracker prunes impossible candidates using motion and coarse geometry. "
                "Pose can then veto clearly incompatible matches when enough keypoints are visible."
            )
        )
        f_assign = QFormLayout(None)
        self._configure_form_layout(f_assign)

        self.spin_assoc_gate_multiplier = QDoubleSpinBox()
        self.spin_assoc_gate_multiplier.setRange(0.5, 5.0)
        self.spin_assoc_gate_multiplier.setDecimals(2)
        self.spin_assoc_gate_multiplier.setSingleStep(0.05)
        self.spin_assoc_gate_multiplier.setValue(1.4)
        self.spin_assoc_gate_multiplier.setToolTip(
            "Multiplier for the stage-1 motion gate before full scoring."
        )

        self.spin_assoc_max_area_ratio = QDoubleSpinBox()
        self.spin_assoc_max_area_ratio.setRange(1.0, 10.0)
        self.spin_assoc_max_area_ratio.setDecimals(2)
        self.spin_assoc_max_area_ratio.setSingleStep(0.1)
        self.spin_assoc_max_area_ratio.setValue(2.5)
        self.spin_assoc_max_area_ratio.setToolTip(
            "Maximum allowed area ratio during candidate gating."
        )

        self.spin_assoc_max_aspect_diff = QDoubleSpinBox()
        self.spin_assoc_max_aspect_diff.setRange(0.0, 5.0)
        self.spin_assoc_max_aspect_diff.setDecimals(2)
        self.spin_assoc_max_aspect_diff.setSingleStep(0.05)
        self.spin_assoc_max_aspect_diff.setValue(0.8)
        self.spin_assoc_max_aspect_diff.setToolTip(
            "Maximum aspect-ratio change allowed during candidate gating."
        )

        f_assign.addRow(
            self._build_field_grid(
                [
                    ("Motion gate ×", self.spin_assoc_gate_multiplier),
                    ("Max area ratio", self.spin_assoc_max_area_ratio),
                    ("Max aspect diff", self.spin_assoc_max_aspect_diff),
                ]
            )
        )

        self.chk_enable_pose_rejection = QCheckBox(
            "Enable pose veto on incompatible same-keypoint layouts"
        )
        self.chk_enable_pose_rejection.setChecked(True)
        self.chk_enable_pose_rejection.setToolTip(
            "Allow pose to veto motion-feasible matches when the same-keypoint layout\n"
            "is clearly incompatible. Thresholds are baked-in defaults\n"
            f"(distance ≤ {POSE_REJECTION_THRESHOLD_CONST}, "
            f"min visibility ≥ {POSE_REJECTION_MIN_VISIBILITY_CONST})."
        )
        f_assign.addRow(self.chk_enable_pose_rejection)

        vl_assign.addLayout(f_assign)
        g_assign.setContentLayout(vl_assign)
        vbox.addWidget(g_assign)
        self._main_window._remember_collapsible_state(
            "tracking.candidate_filtering", g_assign
        )

        # Assignment solver: auto-picked from animal count at runtime
        # (Hungarian for small groups, greedy + spatial indexing above
        # SOLVER_AUTOPICK_GREEDY_THRESHOLD). No UI knob.

        # Identity Decoder — entire section is hidden when identity
        # classification is disabled in the Analyse Individuals panel
        # (see set_identity_section_visible).
        self.g_identity_decoder = CollapsibleGroupBox(
            "How should identity guide assignment?"
        )
        g_identity_decoder = self.g_identity_decoder
        self.tracking_accordion.addCollapsible(g_identity_decoder)
        vl_identity_decoder = QVBoxLayout()
        vl_identity_decoder.addWidget(
            self._main_window._create_help_label(
                "When identity classification is configured, the Bayesian online decoder integrates "
                "CNN and AprilTag evidence into a per-track probability distribution and uses it "
                "as a soft cost term during assignment. The decoder is uncertain in early frames, "
                "so identity influence starts near zero and grows as evidence accumulates."
            )
        )
        f_identity_decoder = QFormLayout(None)
        self._configure_form_layout(f_identity_decoder)

        # ── Master toggle ──────────────────────────────────────────────────
        self.chk_enable_identity_in_tracking = QCheckBox(
            "Use identity to influence tracking"
        )
        self.chk_enable_identity_in_tracking.setChecked(True)
        self.chk_enable_identity_in_tracking.setToolTip(
            "Master switch for identity influence on tracking.\n"
            "When OFF, identity has zero effect on tracking: no online decoder is built,\n"
            "no Bayesian cost term is added, no identity-based rejoin or commit logic runs.\n"
            "Identity classification still runs and labels are still written to the\n"
            "*_with_individual.csv (and offline post-processing such as the fragment\n"
            "solver still works), but the live tracking pipeline is purely geometric."
        )
        self.chk_enable_identity_in_tracking.toggled.connect(
            self._on_identity_in_tracking_toggled
        )
        f_identity_decoder.addRow(self.chk_enable_identity_in_tracking)

        # Subgroup container — hidden as a unit when master toggle is OFF.
        self._identity_subgroup = QWidget()
        sg_layout = QVBoxLayout(self._identity_subgroup)
        sg_layout.setContentsMargins(0, 0, 0, 0)
        sg_layout.setSpacing(4)
        sg_form = QFormLayout()
        self._configure_form_layout(sg_form)
        sg_form.setContentsMargins(0, 0, 0, 0)

        # ── Subgroup 1: assignment cost ────────────────────────────────────
        _hdr_assignment = QLabel("Assignment influence")
        _hdr_assignment.setStyleSheet("font-weight: bold; margin-top: 6px;")
        sg_form.addRow(_hdr_assignment)

        self.chk_enable_identity_online_decoder = QCheckBox(
            "Enable Bayesian identity cost term"
        )
        self.chk_enable_identity_online_decoder.setChecked(False)
        self.chk_enable_identity_online_decoder.setToolTip(
            "Adds a soft identity log-compatibility term to the Hungarian cost matrix\n"
            "during assignment, scaled by the Identity weight below.  When this is OFF,\n"
            "the decoder still maintains beliefs and emits labels, but assignment is\n"
            "driven purely by geometry."
        )
        self.chk_enable_identity_online_decoder.toggled.connect(
            self._on_identity_online_decoder_toggled
        )
        sg_form.addRow(self.chk_enable_identity_online_decoder)

        self.spin_identity_weight = QDoubleSpinBox()
        self.spin_identity_weight.setRange(0.0, 2.0)
        self.spin_identity_weight.setSingleStep(0.05)
        self.spin_identity_weight.setDecimals(2)
        self.spin_identity_weight.setValue(0.3)
        self.spin_identity_weight.setToolTip(
            "Relative weight of identity cost vs. geometric cost.\n"
            "0.0 = identity has no influence (also disables identity-based rejoin).\n"
            "0.3 = nudges Phase-1 association without dominating motion (default).\n"
            "1.0 = balanced with geometry; only raise if the classifier is highly reliable.\n"
            "When the decoder is uncertain this term is near-zero automatically."
        )

        self.spin_identity_rejoin_threshold = QDoubleSpinBox()
        self.spin_identity_rejoin_threshold.setRange(0.0, 1.0)
        self.spin_identity_rejoin_threshold.setSingleStep(0.05)
        self.spin_identity_rejoin_threshold.setDecimals(2)
        self.spin_identity_rejoin_threshold.setValue(0.5)
        self.spin_identity_rejoin_threshold.setToolTip(
            "Minimum identity score for a committed-lost slot to rejoin a detection\n"
            "via identity evidence alone, bypassing the geometric gate."
        )

        # Pack the two cost-term knobs into a compact widget so we can
        # show/hide them as a unit when the Bayesian checkbox toggles.
        self._identity_cost_widget = QWidget()
        _row_id_cost = QHBoxLayout(self._identity_cost_widget)
        _row_id_cost.setContentsMargins(0, 0, 0, 0)
        _row_id_cost.addWidget(QLabel("Identity weight"))
        _row_id_cost.addWidget(self.spin_identity_weight)
        _row_id_cost.addSpacing(6)
        _row_id_cost.addWidget(QLabel("Rejoin threshold"))
        _row_id_cost.addWidget(self.spin_identity_rejoin_threshold)
        sg_form.addRow(self._identity_cost_widget)

        # ── Subgroup 2: belief decoder ────────────────────────────────────
        _hdr_decoder = QLabel("Belief decoder")
        _hdr_decoder.setStyleSheet("font-weight: bold; margin-top: 8px;")
        sg_form.addRow(_hdr_decoder)

        self.spin_identity_commit_threshold = QDoubleSpinBox()
        self.spin_identity_commit_threshold.setRange(0.5, 1.0)
        self.spin_identity_commit_threshold.setSingleStep(0.01)
        self.spin_identity_commit_threshold.setDecimals(2)
        self.spin_identity_commit_threshold.setValue(0.85)
        self.spin_identity_commit_threshold.setToolTip(
            "Posterior confidence required before a track slot commits to an identity.\n"
            "Higher = fewer but more certain identity assignments."
        )

        self.spin_identity_display_threshold = QDoubleSpinBox()
        self.spin_identity_display_threshold.setRange(0.0, 1.0)
        self.spin_identity_display_threshold.setSingleStep(0.05)
        self.spin_identity_display_threshold.setDecimals(2)
        self.spin_identity_display_threshold.setValue(0.6)
        self.spin_identity_display_threshold.setToolTip(
            "Minimum posterior confidence before an identity label is emitted (per-frame).\n"
            "Gates published labels in the live overlay and CSV output."
        )

        sg_form.addRow(
            self._build_field_grid(
                [
                    ("Commit ≥", self.spin_identity_commit_threshold),
                    ("Per-frame label ≥", self.spin_identity_display_threshold),
                ]
            )
        )

        self.spin_identity_transition_epsilon = QDoubleSpinBox()
        self.spin_identity_transition_epsilon.setRange(0.0, 0.1)
        self.spin_identity_transition_epsilon.setSingleStep(0.005)
        self.spin_identity_transition_epsilon.setDecimals(3)
        self.spin_identity_transition_epsilon.setValue(0.02)
        self.spin_identity_transition_epsilon.setToolTip(
            "Off-diagonal probability in the identity Markov transition.\n"
            "Lower = identity is assumed more stable between frames."
        )

        self.spin_identity_unknown_prior = QDoubleSpinBox()
        self.spin_identity_unknown_prior.setRange(0.0, 0.2)
        self.spin_identity_unknown_prior.setSingleStep(0.005)
        self.spin_identity_unknown_prior.setDecimals(3)
        self.spin_identity_unknown_prior.setValue(0.05)
        self.spin_identity_unknown_prior.setToolTip(
            "Prior probability mass reserved for the 'unknown' identity state."
        )

        sg_form.addRow(
            self._build_field_grid(
                [
                    ("Transition ε", self.spin_identity_transition_epsilon),
                    ("Unknown prior", self.spin_identity_unknown_prior),
                ]
            )
        )

        # ── Subgroup 3: live identity-swap correction ─────────────────────
        _hdr_swap = QLabel("Live identity-swap correction")
        _hdr_swap.setStyleSheet("font-weight: bold; margin-top: 8px;")
        sg_form.addRow(_hdr_swap)

        self.chk_enable_identity_swap_correction = QCheckBox(
            "Atomically swap labels on sustained mutual disagreement"
        )
        self.chk_enable_identity_swap_correction.setChecked(True)
        self.chk_enable_identity_swap_correction.setToolTip(
            "When two committed slots show sustained mutual identity disagreement\n"
            "atomically swap their identity labels — trajectories don't move."
        )
        self.chk_enable_identity_swap_correction.toggled.connect(
            self._on_identity_swap_correction_toggled
        )
        sg_form.addRow(self.chk_enable_identity_swap_correction)

        self.spin_identity_swap_min_frames = QSpinBox()
        self.spin_identity_swap_min_frames.setRange(1, 240)
        self.spin_identity_swap_min_frames.setSingleStep(1)
        self.spin_identity_swap_min_frames.setValue(8)
        self.spin_identity_swap_min_frames.setToolTip(
            "Consecutive frames of mutual disagreement required before a swap fires.\n"
            "Lower = catches errors faster but more false swaps."
        )
        # Wrap so we can hide it as a unit when swap correction is OFF.
        self._identity_swap_frames_widget = QWidget()
        _row_swap = QHBoxLayout(self._identity_swap_frames_widget)
        _row_swap.setContentsMargins(0, 0, 0, 0)
        _row_swap.addWidget(QLabel("Swap min frames"))
        _row_swap.addWidget(self.spin_identity_swap_min_frames)
        _row_swap.addStretch(1)
        sg_form.addRow(self._identity_swap_frames_widget)

        sg_layout.addLayout(sg_form)
        f_identity_decoder.addRow(self._identity_subgroup)

        vl_identity_decoder.addLayout(f_identity_decoder)
        g_identity_decoder.setContentLayout(vl_identity_decoder)
        vbox.addWidget(g_identity_decoder)
        self._main_window._remember_collapsible_state(
            "tracking.identity_decoder", g_identity_decoder
        )

        # Orientation & Lifecycle
        g_misc = CollapsibleGroupBox("How should track direction be updated?")
        self.tracking_accordion.addCollapsible(g_misc)
        vl_misc = QVBoxLayout()
        vl_misc.addWidget(
            self._main_window._create_help_label(
                "These settings control the tracked body axis. When pose direction is available it overrides OBB heading; "
                "otherwise movement and smoothing determine how quickly direction can change."
            )
        )
        f_misc = QFormLayout(None)
        self._configure_form_layout(f_misc)

        self.spin_velocity = QDoubleSpinBox()
        self.spin_velocity.setRange(0.1, 100.0)
        self.spin_velocity.setSingleStep(0.5)
        self.spin_velocity.setDecimals(2)
        self.spin_velocity.setValue(5.0)
        self.spin_velocity.setToolTip(
            "Velocity threshold (body-sizes/second) to classify as 'moving'.\n"
            "Below this = stationary (allows larger orientation changes).\n"
            "Above this = moving (instant orientation flip possible).\n"
            "Recommended: 2-10 body-sizes/s."
        )

        self.spin_max_orient = QDoubleSpinBox()
        self.spin_max_orient.setRange(1, 180)
        self.spin_max_orient.setValue(30)
        self.spin_max_orient.setToolTip(
            "Maximum orientation change (degrees) when stationary (1-180).\n"
            "Recommended: 20-45° (prevents orientation jitter)."
        )

        f_misc.addRow(
            self._build_field_grid(
                [
                    ("Moving speed (BL/s)", self.spin_velocity),
                    ("Max stopped Δθ (°)", self.spin_max_orient),
                ]
            )
        )

        self.chk_instant_flip = QCheckBox("Allow instant 180° flips when moving fast")
        self.chk_instant_flip.setChecked(True)
        self.chk_instant_flip.setToolTip(
            "Allow instant 180° orientation flip when moving quickly.\n"
            "Enable for animals that can turn rapidly."
        )
        f_misc.addRow(self.chk_instant_flip)

        self.chk_directed_orient_smoothing = QCheckBox(
            "Consistency check on pose/head-tail flips"
        )
        self.chk_directed_orient_smoothing.setChecked(True)
        self.chk_directed_orient_smoothing.setToolTip(
            "When enabled, 180° flips from directed models (pose / head-tail)\n"
            "are only accepted when motion corroborates the new direction\n"
            "and the detection confidence meets the threshold below.\n"
            "Small changes (≤90°) are always accepted unchanged."
        )
        self.chk_directed_orient_smoothing.toggled.connect(
            self._on_directed_orient_smoothing_toggled
        )
        f_misc.addRow(self.chk_directed_orient_smoothing)

        self.spin_directed_orient_flip_conf = QDoubleSpinBox()
        self.spin_directed_orient_flip_conf.setRange(0.0, 1.0)
        self.spin_directed_orient_flip_conf.setSingleStep(0.05)
        self.spin_directed_orient_flip_conf.setDecimals(2)
        self.spin_directed_orient_flip_conf.setValue(0.7)
        self.spin_directed_orient_flip_conf.setToolTip(
            "Minimum confidence to accept a >90° pose/head-tail flip (0–1).\n"
            "Higher = fewer spurious flips; lower = more responsive."
        )

        self.spin_directed_orient_flip_persist = QSpinBox()
        self.spin_directed_orient_flip_persist.setRange(1, 20)
        self.spin_directed_orient_flip_persist.setValue(3)
        self.spin_directed_orient_flip_persist.setToolTip(
            "Consecutive frames a >90° flip must be observed before it is\n"
            "accepted. Higher values suppress transient classifier errors."
        )

        # Pack flip-conf and flip-persist into one row, hidden together when
        # the smoothing checkbox is OFF.
        self._directed_orient_flip_widget = QWidget()
        _row_flip = QHBoxLayout(self._directed_orient_flip_widget)
        _row_flip.setContentsMargins(0, 0, 0, 0)
        _row_flip.addWidget(QLabel("Flip conf ≥"))
        _row_flip.addWidget(self.spin_directed_orient_flip_conf)
        _row_flip.addSpacing(6)
        _row_flip.addWidget(QLabel("Flip persist (frames)"))
        _row_flip.addWidget(self.spin_directed_orient_flip_persist)
        f_misc.addRow(self._directed_orient_flip_widget)

        # Group the online consistency controls so they can be shown/hidden
        # together when the post-hoc mode is toggled on/off. The flip conf+persist
        # widget is also cascade-hidden by the smoothing checkbox itself.
        self._directed_orient_online_widgets = [
            self.chk_directed_orient_smoothing,
            self._directed_orient_flip_widget,
        ]

        # Note shown in place of the online controls when post-hoc mode is active.
        self.lbl_directed_orient_posthoc_note = self._main_window._create_help_label(
            "Post-hoc global heading consistency is active (head-tail or pose model "
            "detected). Online flip checks are disabled — heading ambiguities are "
            "resolved globally per track in post-processing using a minimum-flips "
            "dynamic-programming algorithm."
        )
        self.lbl_directed_orient_posthoc_note.setVisible(False)
        vl_misc.addWidget(self.lbl_directed_orient_posthoc_note)

        vl_misc.addLayout(f_misc)
        g_misc.setContentLayout(vl_misc)
        vbox.addWidget(g_misc)
        self._main_window._remember_collapsible_state(
            "tracking.direction_updates", g_misc
        )

        # Track Lifecycle
        g_lifecycle = CollapsibleGroupBox("When should tracks be created or dropped?")
        self.tracking_accordion.addCollapsible(g_lifecycle)
        vl_lifecycle = QVBoxLayout()
        vl_lifecycle.addWidget(
            self._main_window._create_help_label(
                "These settings control occlusion tolerance and duplicate-track prevention."
            )
        )
        f_lifecycle = QFormLayout(None)
        self._configure_form_layout(f_lifecycle)

        self.spin_lost_thresh = QDoubleSpinBox()
        self.spin_lost_thresh.setRange(0.01, 10.0)
        self.spin_lost_thresh.setSingleStep(0.05)
        self.spin_lost_thresh.setDecimals(2)
        self.spin_lost_thresh.setValue(0.33)
        self.spin_lost_thresh.setToolTip(
            "Time without detection before track is terminated (seconds).\n"
            "Higher = tracks persist longer during occlusions.\n"
            "Recommended: 0.15-0.70 s."
        )

        self.spin_min_respawn_distance = QDoubleSpinBox()
        self.spin_min_respawn_distance.setRange(0.0, 20.0)
        self.spin_min_respawn_distance.setSingleStep(0.5)
        self.spin_min_respawn_distance.setDecimals(2)
        self.spin_min_respawn_distance.setValue(2.5)
        self.spin_min_respawn_distance.setToolTip(
            "Minimum distance from existing tracks to spawn new track (×body size).\n"
            "Prevents duplicate tracks near existing animals.\n"
            "Recommended: 2-4× body size."
        )

        f_lifecycle.addRow(
            self._build_field_grid(
                [
                    ("Lost threshold (s)", self.spin_lost_thresh),
                    ("Min respawn distance (BL)", self.spin_min_respawn_distance),
                ]
            )
        )
        vl_lifecycle.addLayout(f_lifecycle)
        g_lifecycle.setContentLayout(vl_lifecycle)
        vbox.addWidget(g_lifecycle)
        self._main_window._remember_collapsible_state(
            "tracking.track_lifecycle", g_lifecycle
        )

        # Stability
        g_stab = CollapsibleGroupBox("How strict should track validation be?")
        self.tracking_accordion.addCollapsible(g_stab)
        vl_stab = QVBoxLayout()
        vl_stab.addWidget(
            self._main_window._create_help_label(
                "Use these settings to remove short-lived fragments. The init-counter "
                "minimum is hardcoded to one detection per frame (formerly "
                "Min-detections-to-start, which was confusingly FPS-coupled)."
            )
        )
        f_stab = QFormLayout(None)
        self._configure_form_layout(f_stab)

        self.spin_min_detect = QDoubleSpinBox()
        self.spin_min_detect.setRange(0.01, 30.0)
        self.spin_min_detect.setSingleStep(0.1)
        self.spin_min_detect.setDecimals(2)
        self.spin_min_detect.setValue(0.33)
        self.spin_min_detect.setToolTip(
            "Minimum total detection time to keep a track (seconds).\n"
            "Filters out short-lived false tracks in post-processing.\n"
            "Recommended: 0.15-0.70 s."
        )

        self.spin_min_track = QDoubleSpinBox()
        self.spin_min_track.setRange(0.01, 30.0)
        self.spin_min_track.setSingleStep(0.1)
        self.spin_min_track.setDecimals(2)
        self.spin_min_track.setValue(0.33)
        self.spin_min_track.setToolTip(
            "Minimum tracking time including predictions (seconds).\n"
            "Filters out tracks with too many gaps.\n"
            "Recommended: similar to min detection time."
        )
        f_stab.addRow(
            self._build_field_grid(
                [
                    ("Min detection time (s)", self.spin_min_detect),
                    ("Min total time (s)", self.spin_min_track),
                ]
            )
        )
        vl_stab.addLayout(f_stab)
        g_stab.setContentLayout(vl_stab)
        vbox.addWidget(g_stab)
        self._main_window._remember_collapsible_state("tracking.validation", g_stab)

        # Confidence Density Map
        self.g_density = CollapsibleGroupBox(
            "How should low-confidence density regions be detected?"
        )
        self.tracking_accordion.addCollapsible(self.g_density)
        vl_density = QVBoxLayout()
        vl_density.addWidget(
            self._main_window._create_help_label(
                "Builds a 3-D (x, y, time) confidence density map from the detection cache. "
                "Spatial regions where detections are persistently uncertain are flagged so the "
                "tracker can apply a tighter distance gate there, reducing identity swaps. "
                "Small or short-lived blobs (single-animal artefacts) are suppressed by the "
                "duration and area filters."
            )
        )
        f_density = QFormLayout(None)
        self._configure_form_layout(f_density)

        self.spin_density_temporal_sigma = QDoubleSpinBox()
        self.spin_density_temporal_sigma.setRange(0.5, 10.0)
        self.spin_density_temporal_sigma.setSingleStep(0.5)
        self.spin_density_temporal_sigma.setDecimals(1)
        self.spin_density_temporal_sigma.setValue(2.0)
        self.spin_density_temporal_sigma.setToolTip(
            "Standard deviation (frames) for temporal Gaussian smoothing.\n"
            "Higher values merge nearby low-confidence events.\n"
            "Default: 2.0."
        )

        self.spin_density_conservative_factor = QDoubleSpinBox()
        self.spin_density_conservative_factor.setRange(0.3, 1.0)
        self.spin_density_conservative_factor.setSingleStep(0.05)
        self.spin_density_conservative_factor.setDecimals(2)
        self.spin_density_conservative_factor.setValue(0.70)
        self.spin_density_conservative_factor.setToolTip(
            "Distance gate fraction for detections in flagged density regions.\n"
            "1.0 = disabled, 0.7 = 70% of normal distance. Default: 0.70."
        )

        f_density.addRow(
            self._build_field_grid(
                [
                    ("Temporal sigma (frames)", self.spin_density_temporal_sigma),
                    ("Conservative gate", self.spin_density_conservative_factor),
                ]
            )
        )

        self.spin_density_min_duration = QSpinBox()
        self.spin_density_min_duration.setRange(1, 50)
        self.spin_density_min_duration.setValue(3)
        self.spin_density_min_duration.setToolTip(
            "Minimum temporal duration (frames) for a density region to be kept.\n"
            "Shorter regions usually represent a single isolated animal.\n"
            "Default: 3."
        )

        self.spin_density_min_area_bodies = QDoubleSpinBox()
        self.spin_density_min_area_bodies.setRange(0.0, 10.0)
        self.spin_density_min_area_bodies.setSingleStep(0.05)
        self.spin_density_min_area_bodies.setDecimals(2)
        self.spin_density_min_area_bodies.setValue(0.25)
        self.spin_density_min_area_bodies.setToolTip(
            "Minimum region area in multiples of body area (body_size²).\n"
            "0.25 = at least ¼ of one body area. Default: 0.25."
        )

        f_density.addRow(
            self._build_field_grid(
                [
                    ("Min duration (frames)", self.spin_density_min_duration),
                    ("Min area (body areas)", self.spin_density_min_area_bodies),
                ]
            )
        )

        self.chk_export_confidence_density_video = QCheckBox(
            "Export density diagnostic video"
        )
        self.chk_export_confidence_density_video.setChecked(False)
        self.chk_export_confidence_density_video.setToolTip(
            "Write a reduced-resolution density visualization video alongside\n"
            "the source video. Adds a full extra video write pass — leave off\n"
            "unless you need the diagnostic."
        )
        f_density.addRow(self.chk_export_confidence_density_video)

        vl_density.addLayout(f_density)
        self.g_density.setContentLayout(vl_density)
        vbox.addWidget(self.g_density)
        self._main_window._remember_collapsible_state(
            "tracking.confidence_density", self.g_density
        )

        self.chk_enable_confidence_density_map.stateChanged.connect(
            self._on_confidence_density_map_toggled
        )
        self._on_confidence_density_map_toggled(
            self.chk_enable_confidence_density_map.checkState()
        )

        # Apply initial cascade-hide state so sub-options match toggle defaults.
        self._on_identity_in_tracking_toggled()
        self._on_directed_orient_smoothing_toggled()

        vbox.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll)

    @staticmethod
    def _configure_form_layout(form: QFormLayout) -> None:
        """Apply the compact spacing used by the cleaned-up right-hand panels."""
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

    @staticmethod
    def _build_field_grid(
        fields: list[tuple[str, QWidget]],
        columns: int = 2,
    ) -> QWidget:
        """Arrange labeled controls in compact vertical cells."""
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        for index, (label_text, field_widget) in enumerate(fields):
            row = index // columns
            column = index % columns
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(4)
            label = QLabel(label_text)
            label.setStyleSheet("color: #cccccc;")
            label.setWordWrap(True)
            cell_layout.addWidget(label)
            cell_layout.addWidget(field_widget)
            grid.addWidget(cell, row, column)
        for column in range(columns):
            grid.setColumnStretch(column, 1)
        return widget

    @staticmethod
    def _build_checkbox_grid(
        checkboxes: list[QCheckBox],
        columns: int = 2,
    ) -> QWidget:
        """Arrange related checkboxes in a compact grid."""
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        for index, checkbox in enumerate(checkboxes):
            row = index // columns
            column = index % columns
            grid.addWidget(checkbox, row, column)
        for column in range(columns):
            grid.setColumnStretch(column, 1)
        return widget

    @staticmethod
    def _make_inline_note(text: str) -> QLabel:
        """Create a low-emphasis explanatory note for section intros."""
        label = QLabel(text)
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        label.setStyleSheet("color: #b8b8b8; padding: 0 2px 4px 2px;")
        return label

    def apply_config(self, config: TrackerConfig) -> None:
        """Update panel widgets to reflect a new config object."""
        self._config = config

    def _on_confidence_density_map_toggled(self, state):
        """Show or hide the density-map controls from the top-level tracking toggle."""
        enabled = self.chk_enable_confidence_density_map.isChecked()
        self.g_density.setVisible(enabled)
        self.g_density.setEnabled(enabled)

    def _on_identity_in_tracking_toggled(self, _checked=None):
        """Cascade-hide identity-decoder controls when the master switch is OFF."""
        enabled = self.chk_enable_identity_in_tracking.isChecked()
        self._identity_subgroup.setVisible(enabled)
        if enabled:
            # Re-apply nested cascade rules so widgets restore correct state.
            self._on_identity_online_decoder_toggled()
            self._on_identity_swap_correction_toggled()

    def _on_identity_online_decoder_toggled(self, _checked=None):
        """Hide Identity weight + Rejoin threshold when Bayesian cost term is OFF."""
        on = (
            self.chk_enable_identity_in_tracking.isChecked()
            and self.chk_enable_identity_online_decoder.isChecked()
        )
        self._identity_cost_widget.setVisible(on)

    def _on_identity_swap_correction_toggled(self, _checked=None):
        """Hide Swap min frames when swap correction is OFF."""
        on = (
            self.chk_enable_identity_in_tracking.isChecked()
            and self.chk_enable_identity_swap_correction.isChecked()
        )
        self._identity_swap_frames_widget.setVisible(on)

    def _on_directed_orient_smoothing_toggled(self, _checked=None):
        """Hide flip-confidence and persistence row when smoothing is OFF."""
        on = self.chk_directed_orient_smoothing.isChecked()
        self._directed_orient_flip_widget.setVisible(on)

    def set_identity_section_visible(self, visible: bool) -> None:
        """Hide the entire identity-decoder collapsible.

        Called from the session orchestrator's individual-analysis-mode sync
        whenever identity classification is toggled in the Analyse Individuals
        panel. When identity classification is OFF the entire decoder section
        is irrelevant — hide it rather than leaving disabled controls on
        screen.
        """
        self.g_identity_decoder.setVisible(visible)
        self.g_identity_decoder.setEnabled(visible)

    def sync_directed_orient_posthoc_ui(self, posthoc_active: bool) -> None:
        """Toggle between online-consistency controls and the post-hoc note.

        Called whenever the head-tail / pose model configuration changes so the
        Direction Updates section stays consistent with the active pipeline.

        Args:
            posthoc_active: True when a head-tail or pose model is configured,
                meaning online flip hysteresis is bypassed in favour of the
                global DP step at post-processing time.
        """
        for w in self._directed_orient_online_widgets:
            w.setVisible(not posthoc_active)
            w.setEnabled(not posthoc_active)
        self.lbl_directed_orient_posthoc_note.setVisible(posthoc_active)
