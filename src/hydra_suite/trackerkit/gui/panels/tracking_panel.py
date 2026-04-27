"""TrackingPanel — core tracking parameters, Kalman filter, and assignment config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
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
        self._layout = QVBoxLayout(self)
        self._build_ui()

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

        # Core Params
        g_core = QGroupBox("How should track continuity be handled?")
        self._main_window._set_compact_section_widget(g_core)
        vl_core = QVBoxLayout(g_core)
        vl_core.addWidget(
            self._main_window._create_help_label(
                "These control basic track-to-detection matching. Max movement sets how far an animal can "
                "move between frames. Max speed gates Kalman predictions to physically plausible values. "
                "Recovery search distance helps reconnect lost tracks."
            )
        )
        f_core = QFormLayout(None)
        f_core.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

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
        f_core.addRow("Max movement (body lengths)", self.spin_max_dist)

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
            "Max speed (body lengths/frame)",
            self.spin_kalman_max_velocity,
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
        f_core.addRow("", self.chk_enable_backward)

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
        f_core.addRow("", self.chk_enable_confidence_density_map)
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

        # Create accordion for advanced tracking settings
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
        f_kf.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.spin_kalman_noise = QDoubleSpinBox()
        self.spin_kalman_noise.setRange(0.0, 1.0)
        self.spin_kalman_noise.setDecimals(4)
        self.spin_kalman_noise.setSingleStep(0.001)
        self.spin_kalman_noise.setValue(0.03)
        self.spin_kalman_noise.setToolTip(
            "Process noise covariance (0.0-1.0) for motion prediction.\n"
            "Lower = trust motion model more (smooth, may lag).\n"
            "Higher = trust measurements more (responsive, less smooth).\n"
            "Note: Optimal value depends on frame rate (time step).\n"
            "Recommended: 0.01-0.05 for predictable motion."
        )
        f_kf.addRow("How smooth should motion prediction be?", self.spin_kalman_noise)

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
            "How strongly should detections override prediction?", self.spin_kalman_meas
        )

        self.spin_kalman_damping = QDoubleSpinBox()
        self.spin_kalman_damping.setRange(0.5, 0.999)
        self.spin_kalman_damping.setSingleStep(0.01)
        self.spin_kalman_damping.setDecimals(3)
        self.spin_kalman_damping.setValue(0.95)
        self.spin_kalman_damping.setToolTip(
            "Velocity damping coefficient (0.5-0.99).\n"
            "Controls how quickly velocity decays each frame.\n"
            "Lower = faster decay (better for stop-and-go behavior).\n"
            "Higher = slower decay (better for continuous motion).\n"
            "Recommended: 0.90-0.95"
        )
        f_kf.addRow(
            "How quickly should estimated speed decay?", self.spin_kalman_damping
        )

        # Age-dependent velocity damping
        age_label = QLabel("How conservative should new tracks be?")
        age_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        f_kf.addRow(age_label)

        self.spin_kalman_maturity_age = QDoubleSpinBox()
        self.spin_kalman_maturity_age.setRange(0.01, 2.0)
        self.spin_kalman_maturity_age.setSingleStep(0.02)
        self.spin_kalman_maturity_age.setDecimals(2)
        self.spin_kalman_maturity_age.setValue(0.17)
        self.spin_kalman_maturity_age.setToolTip(
            "Time for a track to reach maturity (seconds).\n"
            "Converted to frames using the acquisition frame rate.\n"
            "Young tracks use conservative velocity estimates.\n"
            "After this time, tracks use full dynamics.\n"
            "Lower = faster adaptation, Higher = more conservative.\n"
            "Recommended: 0.10-0.35 s"
        )
        f_kf.addRow(
            "How long until a track is trusted (seconds)?",
            self.spin_kalman_maturity_age,
        )

        self.spin_kalman_initial_velocity_retention = QDoubleSpinBox()
        self.spin_kalman_initial_velocity_retention.setRange(0.0, 1.0)
        self.spin_kalman_initial_velocity_retention.setSingleStep(0.05)
        self.spin_kalman_initial_velocity_retention.setDecimals(2)
        self.spin_kalman_initial_velocity_retention.setValue(0.2)
        self.spin_kalman_initial_velocity_retention.setToolTip(
            "Initial velocity retention for brand new tracks (0.0-1.0).\n"
            "0.0 = assume stationary (no velocity)\n"
            "1.0 = use full velocity estimate\n"
            "Gradually increases to 1.0 as track ages to maturity.\n"
            "Lower = more conservative (prevents wild predictions).\n"
            "Recommended: 0.1-0.3"
        )
        f_kf.addRow(
            "How much initial speed should new tracks keep?",
            self.spin_kalman_initial_velocity_retention,
        )

        # Anisotropic process noise
        aniso_label = QLabel("Should forward and sideways uncertainty differ?")
        aniso_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        f_kf.addRow(aniso_label)

        self.spin_kalman_longitudinal_noise = QDoubleSpinBox()
        self.spin_kalman_longitudinal_noise.setRange(0.1, 20.0)
        self.spin_kalman_longitudinal_noise.setSingleStep(0.5)
        self.spin_kalman_longitudinal_noise.setDecimals(1)
        self.spin_kalman_longitudinal_noise.setValue(5.0)
        self.spin_kalman_longitudinal_noise.setToolTip(
            "Forward/longitudinal noise multiplier (0.1-20.0).\n"
            "Controls uncertainty in direction of movement.\n"
            "Higher = more uncertainty forward (smoother forward motion).\n"
            "Multiplies base process noise for forward direction.\n"
            "Recommended: 3.0-7.0"
        )
        f_kf.addRow(
            "How much uncertainty along movement direction?",
            self.spin_kalman_longitudinal_noise,
        )

        self.spin_kalman_lateral_noise = QDoubleSpinBox()
        self.spin_kalman_lateral_noise.setRange(0.01, 5.0)
        self.spin_kalman_lateral_noise.setSingleStep(0.05)
        self.spin_kalman_lateral_noise.setDecimals(2)
        self.spin_kalman_lateral_noise.setValue(0.1)
        self.spin_kalman_lateral_noise.setToolTip(
            "Sideways/lateral noise multiplier (0.01-5.0).\n"
            "Controls uncertainty perpendicular to movement.\n"
            "Lower = less uncertainty sideways (constrains lateral drift).\n"
            "Multiplies base process noise for sideways direction.\n"
            "Recommended: 0.05-0.2"
        )
        f_kf.addRow("How much uncertainty sideways?", self.spin_kalman_lateral_noise)

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

        row1 = QHBoxLayout()
        self.spin_Wp = QDoubleSpinBox()
        self.spin_Wp.setRange(0.0, 10.0)
        self.spin_Wp.setValue(1.0)
        self.spin_Wp.setToolTip(
            "Weight for position distance in the assignment cost.\n"
            "Higher = trust spatial proximity more.\n"
            "Recommended: keep this as the dominant term."
        )
        row1.addWidget(QLabel("Position weight"))
        row1.addWidget(self.spin_Wp)

        self.spin_Wo = QDoubleSpinBox()
        self.spin_Wo.setRange(0.0, 10.0)
        self.spin_Wo.setValue(1.0)
        self.spin_Wo.setToolTip(
            "Weight for orientation difference in the assignment cost.\n"
            "Higher = penalize large direction changes more strongly."
        )
        row1.addWidget(QLabel("Direction weight"))
        row1.addWidget(self.spin_Wo)
        l_weights.addLayout(row1)

        row2 = QHBoxLayout()
        self.spin_Wa = QDoubleSpinBox()
        self.spin_Wa.setRange(0.0, 1.0)
        self.spin_Wa.setSingleStep(0.001)
        self.spin_Wa.setDecimals(4)
        self.spin_Wa.setValue(0.001)
        self.spin_Wa.setToolTip(
            "Weight for area difference in the assignment cost.\n"
            "Higher = penalize sudden size changes more strongly."
        )
        row2.addWidget(QLabel("Area weight"))
        row2.addWidget(self.spin_Wa)

        self.spin_Wasp = QDoubleSpinBox()
        self.spin_Wasp.setRange(0.0, 10.0)
        self.spin_Wasp.setValue(0.1)
        self.spin_Wasp.setToolTip(
            "Weight for aspect-ratio difference in the assignment cost.\n"
            "Higher = penalize coarse shape changes more strongly."
        )
        row2.addWidget(QLabel("Aspect weight"))
        row2.addWidget(self.spin_Wasp)
        l_weights.addLayout(row2)

        self.chk_use_mahal = QCheckBox("Use motion-aware distance")
        self.chk_use_mahal.setChecked(True)
        self.chk_use_mahal.setToolTip(
            "Use Mahalanobis distance instead of Euclidean distance for the position term.\n"
            "This makes the matcher respect predicted velocity and uncertainty."
        )
        l_weights.addWidget(self.chk_use_mahal)

        f_w2 = QFormLayout(None)

        self.spin_track_feature_ema_alpha = QDoubleSpinBox()
        self.spin_track_feature_ema_alpha.setRange(0.0, 0.99)
        self.spin_track_feature_ema_alpha.setDecimals(2)
        self.spin_track_feature_ema_alpha.setSingleStep(0.01)
        self.spin_track_feature_ema_alpha.setValue(0.85)
        self.spin_track_feature_ema_alpha.setToolTip(
            "EMA retention for the per-track pose prototype and step-size summary.\n"
            "Higher = slower adaptation (more stable but lags sudden changes).\n"
            "Lower = faster adaptation (more responsive but noisier).\n"
            "Recommended: 0.80-0.95"
        )
        f_w2.addRow("Track feature EMA", self.spin_track_feature_ema_alpha)

        self.spin_assoc_high_conf_threshold = QDoubleSpinBox()
        self.spin_assoc_high_conf_threshold.setRange(0.0, 1.0)
        self.spin_assoc_high_conf_threshold.setDecimals(2)
        self.spin_assoc_high_conf_threshold.setSingleStep(0.05)
        self.spin_assoc_high_conf_threshold.setValue(0.7)
        self.spin_assoc_high_conf_threshold.setToolTip(
            "Minimum detection confidence required before updating the per-track\n"
            "high-confidence step-size summary.\n"
            "Recommended: 0.6-0.8"
        )
        f_w2.addRow("High-conf update threshold", self.spin_assoc_high_conf_threshold)

        l_weights.addLayout(f_w2)
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

        self.spin_assoc_gate_multiplier = QDoubleSpinBox()
        self.spin_assoc_gate_multiplier.setRange(0.5, 5.0)
        self.spin_assoc_gate_multiplier.setDecimals(2)
        self.spin_assoc_gate_multiplier.setSingleStep(0.05)
        self.spin_assoc_gate_multiplier.setValue(1.4)
        self.spin_assoc_gate_multiplier.setToolTip(
            "Multiplier for the stage-1 motion gate before full scoring."
        )
        f_assign.addRow("Motion gate multiplier", self.spin_assoc_gate_multiplier)

        self.spin_assoc_max_area_ratio = QDoubleSpinBox()
        self.spin_assoc_max_area_ratio.setRange(1.0, 10.0)
        self.spin_assoc_max_area_ratio.setDecimals(2)
        self.spin_assoc_max_area_ratio.setSingleStep(0.1)
        self.spin_assoc_max_area_ratio.setValue(2.5)
        self.spin_assoc_max_area_ratio.setToolTip(
            "Maximum allowed area ratio during candidate gating."
        )
        f_assign.addRow("Max area ratio", self.spin_assoc_max_area_ratio)

        self.spin_assoc_max_aspect_diff = QDoubleSpinBox()
        self.spin_assoc_max_aspect_diff.setRange(0.0, 5.0)
        self.spin_assoc_max_aspect_diff.setDecimals(2)
        self.spin_assoc_max_aspect_diff.setSingleStep(0.05)
        self.spin_assoc_max_aspect_diff.setValue(0.8)
        self.spin_assoc_max_aspect_diff.setToolTip(
            "Maximum aspect-ratio change allowed during candidate gating."
        )
        f_assign.addRow("Max aspect diff", self.spin_assoc_max_aspect_diff)

        self.chk_enable_pose_rejection = QCheckBox("Enable pose rejection")
        self.chk_enable_pose_rejection.setChecked(True)
        self.chk_enable_pose_rejection.setToolTip(
            "Allow pose to veto motion-feasible matches when the same-keypoint layout\n"
            "is clearly incompatible."
        )
        f_assign.addRow(self.chk_enable_pose_rejection)

        self.spin_pose_rejection_threshold = QDoubleSpinBox()
        self.spin_pose_rejection_threshold.setRange(0.0, 5.0)
        self.spin_pose_rejection_threshold.setDecimals(2)
        self.spin_pose_rejection_threshold.setSingleStep(0.05)
        self.spin_pose_rejection_threshold.setValue(0.5)
        self.spin_pose_rejection_threshold.setToolTip(
            "Maximum normalized same-keypoint pose distance allowed before rejecting a match.\n"
            "Lower = stricter pose veto."
        )
        f_assign.addRow("Pose rejection threshold", self.spin_pose_rejection_threshold)

        self.spin_pose_rejection_min_visibility = QDoubleSpinBox()
        self.spin_pose_rejection_min_visibility.setRange(0.0, 1.0)
        self.spin_pose_rejection_min_visibility.setDecimals(2)
        self.spin_pose_rejection_min_visibility.setSingleStep(0.05)
        self.spin_pose_rejection_min_visibility.setValue(0.5)
        self.spin_pose_rejection_min_visibility.setToolTip(
            "Minimum pose visibility required before pose rejection is allowed to activate."
        )
        f_assign.addRow(
            "Pose rejection min visibility", self.spin_pose_rejection_min_visibility
        )

        vl_assign.addLayout(f_assign)
        g_assign.setContentLayout(vl_assign)
        vbox.addWidget(g_assign)
        self._main_window._remember_collapsible_state(
            "tracking.candidate_filtering", g_assign
        )

        # Assignment algorithm
        g_solver = CollapsibleGroupBox("Which assignment algorithm should be used?")
        self.tracking_accordion.addCollapsible(g_solver)
        vl_solver = QVBoxLayout()
        vl_solver.addWidget(
            self._main_window._create_help_label(
                "These settings select the core assignment algorithm and whether to use spatial indexing "
                "to speed up matching for larger groups."
            )
        )
        f_solver = QFormLayout(None)

        self.combo_assignment_method = QComboBox()
        self.combo_assignment_method.addItems(
            ["Most accurate (slower)", "Fast approximate (large groups)"]
        )
        self.combo_assignment_method.setCurrentIndex(0)
        self.combo_assignment_method.setToolTip(
            "Hungarian: optimal global assignment.\n"
            "Greedy: faster approximation for very large groups."
        )
        f_solver.addRow("Assignment solver", self.combo_assignment_method)

        self.chk_spatial_optimization = QCheckBox("Speed up matching for many animals")
        self.chk_spatial_optimization.setChecked(False)
        self.chk_spatial_optimization.setToolTip(
            "Use spatial indexing to reduce comparisons when many animals are present.\n"
            "Usually only helpful for larger groups."
        )
        f_solver.addRow(self.chk_spatial_optimization)

        vl_solver.addLayout(f_solver)
        g_solver.setContentLayout(vl_solver)
        vbox.addWidget(g_solver)
        self._main_window._remember_collapsible_state(
            "tracking.assignment_solver", g_solver
        )

        # Identity Decoder
        g_identity_decoder = CollapsibleGroupBox(
            "How should identity guide assignment?"
        )
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
        f_identity_decoder.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.chk_enable_identity_online_decoder = QCheckBox(
            "Enable online identity decoder"
        )
        self.chk_enable_identity_online_decoder.setChecked(False)
        self.chk_enable_identity_online_decoder.setToolTip(
            "Activates the Bayesian online decoder that integrates CNN and AprilTag evidence\n"
            "into the assignment cost as a soft probability term.\n"
            "Requires identity classification to be configured in the Individual Analysis tab."
        )
        f_identity_decoder.addRow(self.chk_enable_identity_online_decoder)

        self.spin_identity_weight = QDoubleSpinBox()
        self.spin_identity_weight.setRange(0.0, 2.0)
        self.spin_identity_weight.setSingleStep(0.05)
        self.spin_identity_weight.setDecimals(2)
        self.spin_identity_weight.setValue(1.0)
        self.spin_identity_weight.setToolTip(
            "Relative weight of the identity cost term vs. the geometric cost.\n"
            "0.0 = identity has no influence on assignment.\n"
            "1.0 = balanced with geometry.\n"
            "When the decoder is uncertain (early frames), this term is near-zero automatically."
        )
        f_identity_decoder.addRow("Identity weight", self.spin_identity_weight)

        self.spin_identity_commit_threshold = QDoubleSpinBox()
        self.spin_identity_commit_threshold.setRange(0.5, 1.0)
        self.spin_identity_commit_threshold.setSingleStep(0.01)
        self.spin_identity_commit_threshold.setDecimals(2)
        self.spin_identity_commit_threshold.setValue(0.85)
        self.spin_identity_commit_threshold.setToolTip(
            "Posterior confidence required before a track slot commits to an identity.\n"
            "Higher = fewer but more certain identity assignments."
        )
        f_identity_decoder.addRow(
            "Commit threshold", self.spin_identity_commit_threshold
        )

        self.spin_identity_display_threshold = QDoubleSpinBox()
        self.spin_identity_display_threshold.setRange(0.0, 1.0)
        self.spin_identity_display_threshold.setSingleStep(0.05)
        self.spin_identity_display_threshold.setDecimals(2)
        self.spin_identity_display_threshold.setValue(0.6)
        self.spin_identity_display_threshold.setToolTip(
            "Minimum posterior confidence before an identity label is shown in the tracking overlay."
        )
        f_identity_decoder.addRow(
            "Display threshold", self.spin_identity_display_threshold
        )

        self.spin_identity_transition_epsilon = QDoubleSpinBox()
        self.spin_identity_transition_epsilon.setRange(0.0, 0.1)
        self.spin_identity_transition_epsilon.setSingleStep(0.005)
        self.spin_identity_transition_epsilon.setDecimals(3)
        self.spin_identity_transition_epsilon.setValue(0.02)
        self.spin_identity_transition_epsilon.setToolTip(
            "Off-diagonal probability in the identity Markov transition.\n"
            "Lower = identity is assumed more stable between frames.\n"
            "Higher = allows faster identity switching."
        )
        f_identity_decoder.addRow(
            "Transition epsilon", self.spin_identity_transition_epsilon
        )

        self.spin_identity_unknown_prior = QDoubleSpinBox()
        self.spin_identity_unknown_prior.setRange(0.0, 0.2)
        self.spin_identity_unknown_prior.setSingleStep(0.005)
        self.spin_identity_unknown_prior.setDecimals(3)
        self.spin_identity_unknown_prior.setValue(0.05)
        self.spin_identity_unknown_prior.setToolTip(
            "Prior probability mass reserved for the 'unknown' identity state.\n"
            "Higher = more uncertainty about whether any known identity applies."
        )
        f_identity_decoder.addRow("Unknown prior", self.spin_identity_unknown_prior)

        self.spin_identity_rejoin_threshold = QDoubleSpinBox()
        self.spin_identity_rejoin_threshold.setRange(0.0, 1.0)
        self.spin_identity_rejoin_threshold.setSingleStep(0.05)
        self.spin_identity_rejoin_threshold.setDecimals(2)
        self.spin_identity_rejoin_threshold.setValue(0.5)
        self.spin_identity_rejoin_threshold.setToolTip(
            "Minimum identity score (probability) for a committed-lost slot to rejoin "
            "a detection via identity evidence alone, bypassing the geometric gate."
        )
        f_identity_decoder.addRow(
            "Rejoin threshold", self.spin_identity_rejoin_threshold
        )

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

        self.spin_velocity = QDoubleSpinBox()
        self.spin_velocity.setRange(0.1, 100.0)
        self.spin_velocity.setSingleStep(0.5)
        self.spin_velocity.setDecimals(2)
        self.spin_velocity.setValue(5.0)
        self.spin_velocity.setToolTip(
            "Velocity threshold (body-sizes/second) to classify as 'moving'.\n"
            "Below this = stationary (allows larger orientation changes).\n"
            "Above this = moving (instant orientation flip possible).\n"
            "Independent of frame rate - automatically scaled by FPS.\n"
            "Recommended: 2-10 body-sizes/s depending on animal speed."
        )
        f_misc.addRow("Moving-speed threshold (body lengths/sec)", self.spin_velocity)

        self.chk_instant_flip = QCheckBox(
            "Allow instant direction flips when moving fast"
        )
        self.chk_instant_flip.setChecked(True)
        self.chk_instant_flip.setToolTip(
            "Allow instant 180° orientation flip when moving quickly.\n"
            "Recommended: Enable for animals that can turn rapidly.\n"
            "Disable for slowly rotating animals."
        )
        f_misc.addRow(self.chk_instant_flip)

        self.spin_max_orient = QDoubleSpinBox()
        self.spin_max_orient.setRange(1, 180)
        self.spin_max_orient.setValue(30)
        self.spin_max_orient.setToolTip(
            "Maximum orientation change (degrees) when stationary (1-180).\n"
            "Larger = allow more rotation while stopped.\n"
            "Recommended: 20-45° (prevents orientation jitter)."
        )
        f_misc.addRow(
            "Max direction change while stopped (degrees)", self.spin_max_orient
        )

        self.chk_directed_orient_smoothing = QCheckBox(
            "Apply consistency check to pose/head-tail orientation flips"
        )
        self.chk_directed_orient_smoothing.setChecked(True)
        self.chk_directed_orient_smoothing.setToolTip(
            "When enabled, 180° flips from directed models (pose / head-tail)\n"
            "are only accepted when motion corroborates the new direction\n"
            "and the detection confidence meets the threshold below.\n"
            "Small changes (≤90°) are always accepted unchanged."
        )
        f_misc.addRow(self.chk_directed_orient_smoothing)

        self.spin_directed_orient_flip_conf = QDoubleSpinBox()
        self.spin_directed_orient_flip_conf.setRange(0.0, 1.0)
        self.spin_directed_orient_flip_conf.setSingleStep(0.05)
        self.spin_directed_orient_flip_conf.setDecimals(2)
        self.spin_directed_orient_flip_conf.setValue(0.7)
        self.spin_directed_orient_flip_conf.setToolTip(
            "Minimum confidence to accept a >90° pose/head-tail orientation flip (0–1).\n"
            "For pose-directed headings, this is the pose visibility score.\n"
            "For head-tail-directed headings, this is the classifier confidence.\n"
            "Higher = fewer spurious flips; lower = more responsive."
        )
        f_misc.addRow(
            "Directed-flip confidence threshold", self.spin_directed_orient_flip_conf
        )

        self.spin_directed_orient_flip_persist = QSpinBox()
        self.spin_directed_orient_flip_persist.setRange(1, 20)
        self.spin_directed_orient_flip_persist.setValue(3)
        self.spin_directed_orient_flip_persist.setToolTip(
            "Number of consecutive frames a >90° heading flip must be observed\n"
            "before it is accepted as genuine. Higher values suppress transient\n"
            "head-tail classifier errors at the cost of slower real-turn response."
        )
        f_misc.addRow(
            "Directed-flip persistence (frames)", self.spin_directed_orient_flip_persist
        )

        # Group the three online consistency controls so they can be shown/hidden
        # together when the post-hoc mode is toggled on/off.
        self._directed_orient_online_widgets = [
            self.chk_directed_orient_smoothing,
            self.spin_directed_orient_flip_conf,
            self.spin_directed_orient_flip_persist,
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

        self.spin_lost_thresh = QDoubleSpinBox()
        self.spin_lost_thresh.setRange(0.01, 10.0)
        self.spin_lost_thresh.setSingleStep(0.05)
        self.spin_lost_thresh.setDecimals(2)
        self.spin_lost_thresh.setValue(0.33)
        self.spin_lost_thresh.setToolTip(
            "Time without detection before track is terminated (seconds).\n"
            "Converted to frames using the acquisition frame rate.\n"
            "Higher = tracks persist longer during occlusions.\n"
            "Lower = tracks end quickly, creating fragments.\n"
            "Recommended: 0.15-0.70 s."
        )
        f_lifecycle.addRow(
            "How long to keep a track without detections (seconds)?",
            self.spin_lost_thresh,
        )

        self.spin_min_respawn_distance = QDoubleSpinBox()
        self.spin_min_respawn_distance.setRange(0.0, 20.0)
        self.spin_min_respawn_distance.setSingleStep(0.5)
        self.spin_min_respawn_distance.setDecimals(2)
        self.spin_min_respawn_distance.setValue(2.5)
        self.spin_min_respawn_distance.setToolTip(
            "Minimum distance from existing tracks to spawn new track (×body size).\n"
            "Prevents creating duplicate tracks near existing animals.\n"
            "Recommended: 2-4× body size."
        )
        f_lifecycle.addRow(
            "How far from existing tracks to start a new one (body lengths)?",
            self.spin_min_respawn_distance,
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
                "Use these settings to suppress noisy starts and remove short-lived fragments."
            )
        )
        f_stab = QFormLayout(None)
        self.spin_min_detections_to_start = QDoubleSpinBox()
        self.spin_min_detections_to_start.setRange(0.01, 2.0)
        self.spin_min_detections_to_start.setSingleStep(0.02)
        self.spin_min_detections_to_start.setDecimals(2)
        self.spin_min_detections_to_start.setValue(0.03)
        self.spin_min_detections_to_start.setToolTip(
            "Minimum time of consecutive detections before starting a track (seconds).\n"
            "Converted to frames using the acquisition frame rate.\n"
            "Higher = fewer false tracks from noise, slower to start tracking.\n"
            "Lower = faster tracking startup, more noise-based tracks.\n"
            "Recommended: 0.03-0.10 s"
        )
        f_stab.addRow(
            "How long must detections persist before starting a track (seconds)?",
            self.spin_min_detections_to_start,
        )

        self.spin_min_detect = QDoubleSpinBox()
        self.spin_min_detect.setRange(0.01, 30.0)
        self.spin_min_detect.setSingleStep(0.1)
        self.spin_min_detect.setDecimals(2)
        self.spin_min_detect.setValue(0.33)
        self.spin_min_detect.setToolTip(
            "Minimum total detection time to keep a track (seconds).\n"
            "Converted to frames using the acquisition frame rate.\n"
            "Filters out short-lived false tracks in post-processing.\n"
            "Recommended: 0.15-0.70 s."
        )

        self.spin_min_track = QDoubleSpinBox()
        self.spin_min_track.setRange(0.01, 30.0)
        self.spin_min_track.setSingleStep(0.1)
        self.spin_min_track.setDecimals(2)
        self.spin_min_track.setValue(0.33)
        self.spin_min_track.setToolTip(
            "Minimum tracking time (including predicted) to keep (seconds).\n"
            "Converted to frames using the acquisition frame rate.\n"
            "Filters out tracks with too many gaps/predictions.\n"
            "Recommended: similar to min detection time."
        )
        _min_frames_row = QHBoxLayout()
        _min_frames_row.addWidget(QLabel("Min detection time (s)"))
        _min_frames_row.addWidget(self.spin_min_detect)
        _min_frames_row.addWidget(QLabel("Min total time (s)"))
        _min_frames_row.addWidget(self.spin_min_track)
        f_stab.addRow(_min_frames_row)
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
        f_density.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.spin_density_gaussian_sigma_scale = QDoubleSpinBox()
        self.spin_density_gaussian_sigma_scale.setRange(0.1, 5.0)
        self.spin_density_gaussian_sigma_scale.setSingleStep(0.1)
        self.spin_density_gaussian_sigma_scale.setDecimals(1)
        self.spin_density_gaussian_sigma_scale.setValue(1.0)
        self.spin_density_gaussian_sigma_scale.setToolTip(
            "Scale factor for the Gaussian sigma derived from detection size.\n"
            "Controls the spatial spread of each detection's contribution\n"
            "to the confidence density map. Larger values produce smoother maps.\n"
            "Range: 0.1–5.0. Default: 1.0."
        )
        f_density.addRow("Gaussian sigma scale", self.spin_density_gaussian_sigma_scale)

        self.spin_density_temporal_sigma = QDoubleSpinBox()
        self.spin_density_temporal_sigma.setRange(0.5, 10.0)
        self.spin_density_temporal_sigma.setSingleStep(0.5)
        self.spin_density_temporal_sigma.setDecimals(1)
        self.spin_density_temporal_sigma.setValue(2.0)
        self.spin_density_temporal_sigma.setToolTip(
            "Standard deviation (in frames) for temporal Gaussian smoothing\n"
            "of the confidence density volume. Higher values merge nearby\n"
            "low-confidence events into broader temporal regions.\n"
            "Range: 0.5–10.0. Default: 2.0."
        )
        f_density.addRow(
            "Temporal smoothing sigma (frames)", self.spin_density_temporal_sigma
        )

        self.spin_density_binarize_threshold = QDoubleSpinBox()
        self.spin_density_binarize_threshold.setRange(0.05, 0.95)
        self.spin_density_binarize_threshold.setSingleStep(0.05)
        self.spin_density_binarize_threshold.setDecimals(2)
        self.spin_density_binarize_threshold.setValue(0.3)
        self.spin_density_binarize_threshold.setToolTip(
            "Threshold for binarizing the normalised density volume.\n"
            "Voxels above this value become foreground regions where\n"
            "density-aware conservative tracking is applied.\n"
            "Range: 0.05–0.95. Default: 0.3."
        )
        f_density.addRow("Binarize threshold", self.spin_density_binarize_threshold)

        self.spin_density_conservative_factor = QDoubleSpinBox()
        self.spin_density_conservative_factor.setRange(0.3, 1.0)
        self.spin_density_conservative_factor.setSingleStep(0.05)
        self.spin_density_conservative_factor.setDecimals(2)
        self.spin_density_conservative_factor.setValue(0.70)
        self.spin_density_conservative_factor.setToolTip(
            "Distance gate fraction for detections in flagged density regions.\n"
            "Reduces the maximum assignment distance for in-region detections\n"
            "to prevent long-range jumps into crowded zones.\n"
            "1.0 = disabled, 0.7 = 70% of normal distance.\n"
            "Range: 0.3–1.0. Default: 0.70."
        )
        f_density.addRow(
            "Conservative distance gate", self.spin_density_conservative_factor
        )

        self.spin_density_min_duration = QSpinBox()
        self.spin_density_min_duration.setRange(1, 50)
        self.spin_density_min_duration.setValue(3)
        self.spin_density_min_duration.setToolTip(
            "Minimum temporal duration (frames) for a density region to be kept.\n"
            "Regions shorter than this are discarded — they usually represent a\n"
            "single isolated animal rather than a genuine crowding event.\n"
            "Range: 1–50. Default: 3."
        )
        f_density.addRow("Min region duration (frames)", self.spin_density_min_duration)

        self.spin_density_min_area_bodies = QDoubleSpinBox()
        self.spin_density_min_area_bodies.setRange(0.0, 10.0)
        self.spin_density_min_area_bodies.setSingleStep(0.05)
        self.spin_density_min_area_bodies.setDecimals(2)
        self.spin_density_min_area_bodies.setValue(0.25)
        self.spin_density_min_area_bodies.setToolTip(
            "Minimum spatial area of a density region expressed as multiples of\n"
            "the reference body area (body_size²). Regions smaller than this in\n"
            "the density grid are discarded as single-animal artefacts.\n"
            "E.g. 0.25 requires the region to cover at least ¼ of one body area.\n"
            "Range: 0.0–10.0. Default: 0.25."
        )
        f_density.addRow(
            "Min region area (body areas)", self.spin_density_min_area_bodies
        )

        self.spin_density_downsample_factor = QSpinBox()
        self.spin_density_downsample_factor.setRange(1, 32)
        self.spin_density_downsample_factor.setValue(8)
        self.spin_density_downsample_factor.setToolTip(
            "Spatial downsampling factor applied to the density grid.\n"
            "Higher values make computation faster but reduce spatial precision.\n"
            "The grid will be (frame_h / factor) × (frame_w / factor).\n"
            "Range: 1–32. Default: 8."
        )
        f_density.addRow("Grid downsample factor", self.spin_density_downsample_factor)

        self.chk_export_confidence_density_video = QCheckBox(
            "Export density diagnostic video"
        )
        self.chk_export_confidence_density_video.setChecked(False)
        self.chk_export_confidence_density_video.setToolTip(
            "Write a reduced-resolution confidence-density visualization video\n"
            "next to the source video after density-map computation completes.\n"
            "Leave disabled unless you need the diagnostic visualization,\n"
            "because exporting it adds a full extra video write pass."
        )
        f_density.addRow("", self.chk_export_confidence_density_video)

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

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def apply_config(self, config: TrackerConfig) -> None:
        """Update panel widgets to reflect a new config object."""
        self._config = config

    def _on_confidence_density_map_toggled(self, state):
        """Show or hide the density-map controls from the top-level tracking toggle."""
        enabled = self.chk_enable_confidence_density_map.isChecked()
        self.g_density.setVisible(enabled)
        self.g_density.setEnabled(enabled)

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
