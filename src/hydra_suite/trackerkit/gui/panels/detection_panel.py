"""DetectionPanel — detection method, image preprocessing, and model config."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.trackerkit.config.schemas import TrackerConfig
from hydra_suite.utils.batch_policy import is_realtime_workflow
from hydra_suite.utils.gpu_utils import (
    MPS_AVAILABLE,
    TENSORRT_AVAILABLE,
    TORCH_CUDA_AVAILABLE,
)

if TYPE_CHECKING:
    from hydra_suite.trackerkit.gui.main_window import MainWindow

logger = logging.getLogger(__name__)


class DetectionPanel(QWidget):
    """Detection method selector, image-processing pipeline, and YOLO config."""

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
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._build_ui()

    def _build_ui(self) -> None:
        """Populate the panel layout."""
        from hydra_suite.trackerkit.gui.widgets.collapsible import (
            AccordionContainer,
            CollapsibleGroupBox,
        )

        layout = self._layout
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content = QWidget()
        vbox = QVBoxLayout(content)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(8)
        self._main_window._set_compact_scroll_layout(vbox)

        # ============================================================
        # 1. Detection Method Selector
        # ============================================================
        g_method = QGroupBox("Detection")
        self._main_window._set_compact_section_widget(g_method)
        l_method_outer = QVBoxLayout(g_method)
        l_method_outer.setSpacing(6)
        f_method = QFormLayout(None)
        f_method.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        f_method.setHorizontalSpacing(10)
        f_method.setVerticalSpacing(8)
        method_help = self._main_window._create_help_label(
            "Choose how to detect animals in each frame. Background subtraction models the static background and finds moving objects. YOLO uses deep learning to detect animals directly.",
            attach_to_title=False,
        )
        self.combo_detection_method = QComboBox()
        self.combo_detection_method.addItems(["Background Subtraction", "YOLO OBB"])
        self.combo_detection_method.setFixedHeight(30)
        self.combo_detection_method.currentIndexChanged.connect(
            self._on_detection_method_changed_ui
        )
        method_row = QHBoxLayout()
        method_row.setSpacing(6)
        method_row.addWidget(self.combo_detection_method, 1)
        method_row.addWidget(method_help, 0, Qt.AlignVCenter)
        f_method.addRow("Method", method_row)

        # Legacy device selection (hidden; derived from canonical runtime).
        self.combo_device = QComboBox()
        device_options = ["auto", "cpu"]
        device_tooltip_parts = [
            "Select compute device for detection:",
            "  • auto - Automatically select best available device",
            "  • cpu - CPU-only mode",
        ]

        if TORCH_CUDA_AVAILABLE:
            device_options.append("cuda:0")
            device_tooltip_parts.append("  • cuda:0 - NVIDIA GPU ✓ Available")
        else:
            device_tooltip_parts.append("  • cuda:0 - NVIDIA GPU (not available)")

        if MPS_AVAILABLE:
            device_options.append("mps")
            device_tooltip_parts.append("  • mps - Apple Silicon GPU ✓ Available")
        else:
            device_tooltip_parts.append("  • mps - Apple Silicon GPU (not available)")

        device_tooltip_parts.append(
            "\nApplies to both YOLO and Background Subtraction GPU acceleration."
        )

        self.combo_device.addItems(device_options)
        self.combo_device.setToolTip("\n".join(device_tooltip_parts))
        f_method.addRow("Which compute device should run detection?", self.combo_device)
        device_label = f_method.labelForField(self.combo_device)
        if device_label is not None:
            device_label.setVisible(False)
        self.combo_device.setVisible(False)

        l_method_outer.addLayout(f_method)
        vbox.addWidget(g_method)

        # Stacked Widget for Method Specific Params
        self.stack_detection = QStackedWidget()
        self.stack_detection.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        # --- Page 0: Background Subtraction Params ---
        page_bg = QWidget()
        page_bg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        l_bg = QVBoxLayout(page_bg)
        l_bg.setContentsMargins(0, 0, 0, 0)
        l_bg.setSpacing(6)

        # Create accordion for BG subtraction settings
        self.bg_accordion = AccordionContainer()

        # Image Enhancement (Pre-processing)
        self.g_img = CollapsibleGroupBox("Image")
        self.bg_accordion.addCollapsible(self.g_img)
        vl_img = QVBoxLayout()
        vl_img.addWidget(
            self._main_window._create_help_label(
                "Adjust image properties before detection to improve contrast between animals and background. "
                "Start with default values and adjust only if animals are hard to distinguish."
            )
        )

        # Brightness slider
        bright_layout = QVBoxLayout()
        bright_label_row = QHBoxLayout()
        bright_label_row.addWidget(QLabel("Brightness"))
        self.label_brightness_val = QLabel("0")
        self.label_brightness_val.setStyleSheet("color: #4fc1ff; font-weight: bold;")
        bright_label_row.addWidget(self.label_brightness_val)
        bright_label_row.addSpacing(6)
        bright_layout.addLayout(bright_label_row)

        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setRange(-255, 255)
        self.slider_brightness.setValue(0)
        self.slider_brightness.setTickPosition(QSlider.TicksBelow)
        self.slider_brightness.setTickInterval(50)
        self.slider_brightness.valueChanged.connect(self._on_brightness_changed)
        self.slider_brightness.setToolTip(
            "Adjust overall image brightness.\n"
            "Positive = lighter, Negative = darker.\n"
            "Use to improve contrast between animals and background."
        )
        bright_layout.addWidget(self.slider_brightness)
        vl_img.addLayout(bright_layout)

        # Contrast slider
        contrast_layout = QVBoxLayout()
        contrast_label_row = QHBoxLayout()
        contrast_label_row.addWidget(QLabel("Contrast"))
        self.label_contrast_val = QLabel("1.0")
        self.label_contrast_val.setStyleSheet("color: #4fc1ff; font-weight: bold;")
        contrast_label_row.addWidget(self.label_contrast_val)
        contrast_label_row.addSpacing(6)
        contrast_layout.addLayout(contrast_label_row)

        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setRange(0, 300)  # 0.0 to 3.0, scaled by 100
        self.slider_contrast.setValue(100)  # 1.0
        self.slider_contrast.setTickPosition(QSlider.TicksBelow)
        self.slider_contrast.setTickInterval(50)
        self.slider_contrast.valueChanged.connect(self._on_contrast_changed)
        self.slider_contrast.setToolTip(
            "Adjust image contrast (difference between light and dark).\n"
            "1.0 = original, >1.0 = more contrast, <1.0 = less contrast.\n"
            "Increase to make animals stand out from background."
        )
        contrast_layout.addWidget(self.slider_contrast)
        vl_img.addLayout(contrast_layout)

        # Gamma slider
        gamma_layout = QVBoxLayout()
        gamma_label_row = QHBoxLayout()
        gamma_label_row.addWidget(QLabel("Gamma"))
        self.label_gamma_val = QLabel("1.0")
        self.label_gamma_val.setStyleSheet("color: #4fc1ff; font-weight: bold;")
        gamma_label_row.addWidget(self.label_gamma_val)
        gamma_label_row.addSpacing(6)
        gamma_layout.addLayout(gamma_label_row)

        self.slider_gamma = QSlider(Qt.Horizontal)
        self.slider_gamma.setRange(10, 300)  # 0.1 to 3.0, scaled by 100
        self.slider_gamma.setValue(100)  # 1.0
        self.slider_gamma.setTickPosition(QSlider.TicksBelow)
        self.slider_gamma.setTickInterval(50)
        self.slider_gamma.valueChanged.connect(self._on_gamma_changed)
        self.slider_gamma.setToolTip(
            "Adjust gamma correction (mid-tone brightness).\n"
            "1.0 = original, >1.0 = brighter mid-tones, <1.0 = darker mid-tones.\n"
            "Use to enhance detail in shadowed or bright areas."
        )
        gamma_layout.addWidget(self.slider_gamma)
        vl_img.addLayout(gamma_layout)

        # Dark on light checkbox
        self.chk_dark_on_light = QCheckBox("Animals are darker than background")
        self.chk_dark_on_light.setChecked(True)
        self.chk_dark_on_light.setToolTip(
            "Check if animals are darker than background (most common).\n"
            "Uncheck if animals are lighter than background.\n"
            "This inverts the foreground detection."
        )
        vl_img.addWidget(self.chk_dark_on_light)
        self.g_img.setContentLayout(vl_img)
        l_bg.addWidget(self.g_img)
        self._main_window._remember_collapsible_state(
            "detection.brightness_contrast_gamma", self.g_img
        )

        # Background Model
        g_bg_model = CollapsibleGroupBox("Background")
        self.bg_accordion.addCollapsible(g_bg_model)
        vl_bg_model = QVBoxLayout()
        vl_bg_model.addWidget(
            self._main_window._create_help_label(
                "Build a model of the static background. Priming frames establish initial model, learning rate "
                "controls adaptation speed, threshold sets sensitivity. Lower threshold = more sensitive detection."
            )
        )
        f_bg = QFormLayout(None)
        f_bg.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        f_bg.setHorizontalSpacing(10)
        f_bg.setVerticalSpacing(8)
        self.spin_bg_prime = QDoubleSpinBox()
        self.spin_bg_prime.setRange(0.0, 120.0)
        self.spin_bg_prime.setSingleStep(0.5)
        self.spin_bg_prime.setDecimals(2)
        self.spin_bg_prime.setValue(0.33)
        self.spin_bg_prime.setFixedHeight(30)
        self.spin_bg_prime.setToolTip(
            "Time to build background model (seconds).\n"
            "Converted to frames using the acquisition frame rate.\n"
            "Recommended: 0.3-3.0 s.\n"
            "Use more if background varies or animals are present initially."
        )
        f_bg.addRow("Startup time (seconds)", self.spin_bg_prime)

        self.chk_adaptive_bg = QCheckBox("Continuously update background model")
        self.chk_adaptive_bg.setChecked(True)
        self.chk_adaptive_bg.setToolTip(
            "Continuously update background model during tracking.\n"
            "Recommended: Enable for videos with changing lighting.\n"
            "Disable for static background to improve performance."
        )
        f_bg.addRow(self.chk_adaptive_bg)

        self.spin_bg_learning = QDoubleSpinBox()
        self.spin_bg_learning.setRange(0.0001, 0.1)
        self.spin_bg_learning.setDecimals(4)
        self.spin_bg_learning.setValue(0.001)
        self.spin_bg_learning.setFixedHeight(30)
        self.spin_bg_learning.setToolTip(
            "How quickly background adapts to changes (0.0001-0.1).\n"
            "Lower = slower adaptation (stable, good for mostly static background).\n"
            "Higher = faster adaptation (use for variable lighting/shadows)."
        )
        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(0, 255)
        self.spin_threshold.setValue(50)
        self.spin_threshold.setFixedHeight(30)
        self.spin_threshold.setToolTip(
            "Pixel intensity difference to detect foreground (0-255).\n"
            "Lower = more sensitive (detects subtle animals, more noise).\n"
            "Higher = less sensitive (cleaner, may miss animals).\n"
            "Recommended: 30-70 depending on contrast."
        )
        _bg_rate_row = QHBoxLayout()
        _bg_rate_row.addWidget(QLabel("Learn rate:"))
        _bg_rate_row.addWidget(self.spin_bg_learning, 1)
        _bg_rate_row.addSpacing(8)
        _bg_rate_row.addWidget(QLabel("Threshold:"))
        _bg_rate_row.addWidget(self.spin_threshold, 1)
        f_bg.addRow(_bg_rate_row)
        vl_bg_model.addLayout(f_bg)
        g_bg_model.setContentLayout(vl_bg_model)
        l_bg.addWidget(g_bg_model)
        self._main_window._remember_collapsible_state(
            "detection.background_estimation", g_bg_model
        )

        # Lighting Stab
        g_light = CollapsibleGroupBox("Lighting")
        self.bg_accordion.addCollapsible(g_light)
        vl_light = QVBoxLayout()
        vl_light.addWidget(
            self._main_window._create_help_label(
                "Compensate for gradual lighting changes (clouds, time of day). Smoothing factor controls "
                "adaptation speed - higher = slower/more stable. Enable for outdoor or variable-light videos."
            )
        )
        f_light = QFormLayout(None)
        f_light.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        f_light.setHorizontalSpacing(10)
        f_light.setVerticalSpacing(8)
        self.chk_lighting_stab = QCheckBox("Enable Stabilization")
        self.chk_lighting_stab.setChecked(True)
        self.chk_lighting_stab.setToolTip(
            "Compensate for gradual lighting changes over time.\n"
            "Recommended: Enable for videos with variable lighting.\n"
            "Disable for consistent illumination to improve speed."
        )
        f_light.addRow(self.chk_lighting_stab)

        self.spin_lighting_smooth = QDoubleSpinBox()
        self.spin_lighting_smooth.setRange(0.8, 0.999)
        self.spin_lighting_smooth.setValue(0.95)
        self.spin_lighting_smooth.setFixedHeight(30)
        self.spin_lighting_smooth.setToolTip(
            "Temporal smoothing factor for lighting correction (0.8-0.999).\n"
            "Higher = smoother, slower adaptation to lighting changes.\n"
            "Lower = faster response to sudden lighting shifts.\n"
            "Recommended: 0.9-0.98"
        )
        self.spin_lighting_median = QSpinBox()
        self.spin_lighting_median.setRange(3, 15)
        self.spin_lighting_median.setSingleStep(2)
        self.spin_lighting_median.setValue(5)
        self.spin_lighting_median.setFixedHeight(30)
        self.spin_lighting_median.setToolTip(
            "Median filter window size (odd number, 3-15).\n"
            "Larger window = smoother lighting estimate, slower response.\n"
            "Smaller window = faster response, less smoothing.\n"
            "Recommended: 5-9"
        )
        _light_row = QHBoxLayout()
        _light_row.addWidget(QLabel("Smoothing:"))
        _light_row.addWidget(self.spin_lighting_smooth, 1)
        _light_row.addSpacing(8)
        _light_row.addWidget(QLabel("Median (frames):"))
        _light_row.addWidget(self.spin_lighting_median, 1)
        f_light.addRow(_light_row)
        vl_light.addLayout(f_light)
        g_light.setContentLayout(vl_light)
        l_bg.addWidget(g_light)
        self._main_window._remember_collapsible_state(
            "detection.scene_lighting", g_light
        )

        # Morphology (Standard)
        g_morph = CollapsibleGroupBox("Morphology")
        self.bg_accordion.addCollapsible(g_morph)
        vl_morph = QVBoxLayout()
        vl_morph.addWidget(
            self._main_window._create_help_label(
                "Clean up detected blobs using morphological operations. Closing fills small holes, opening removes "
                "small noise. Larger kernels = stronger effect but may distort shape. Use odd numbers only."
            )
        )
        f_morph = QFormLayout(None)
        f_morph.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        f_morph.setHorizontalSpacing(10)
        f_morph.setVerticalSpacing(8)
        self.spin_morph_size = QSpinBox()
        self.spin_morph_size.setRange(1, 25)
        self.spin_morph_size.setSingleStep(2)
        self.spin_morph_size.setValue(5)
        self.spin_morph_size.setFixedHeight(30)
        self.spin_morph_size.setToolTip(
            "Morphological operation kernel size (odd number, 1-25).\n"
            "Larger = more aggressive noise removal, may merge nearby animals.\n"
            "Smaller = preserves detail, may leave noise.\n"
            "Recommended: 3-7 for typical tracking scenarios."
        )
        f_morph.addRow("Kernel size", self.spin_morph_size)

        self.spin_min_contour = QSpinBox()
        self.spin_min_contour.setRange(0, 100000)
        self.spin_min_contour.setValue(50)
        self.spin_min_contour.setFixedHeight(30)
        self.spin_min_contour.setToolTip(
            "Minimum contour area in pixels² to keep.\n"
            "Filters out small noise blobs after morphology.\n"
            "Recommended: 20-100 depending on animal size and zoom.\n"
            "Note: Similar to min object size but in absolute pixels."
        )
        self.spin_max_contour_multiplier = QSpinBox()
        self.spin_max_contour_multiplier.setRange(5, 100)
        self.spin_max_contour_multiplier.setValue(20)
        self.spin_max_contour_multiplier.setFixedHeight(30)
        self.spin_max_contour_multiplier.setToolTip(
            "Maximum contour area as multiplier of minimum (5-100).\n"
            "Max area = min_contour × this multiplier.\n"
            "Filters out very large blobs (clusters, shadows, artifacts).\n"
            "Recommended: 10-30"
        )
        _contour_row = QHBoxLayout()
        _contour_row.addWidget(QLabel("Min area (px²):"))
        _contour_row.addWidget(self.spin_min_contour, 1)
        _contour_row.addSpacing(8)
        _contour_row.addWidget(QLabel("Max multiplier:"))
        _contour_row.addWidget(self.spin_max_contour_multiplier, 1)
        f_morph.addRow(_contour_row)
        vl_morph.addLayout(f_morph)
        g_morph.setContentLayout(vl_morph)
        l_bg.addWidget(g_morph)
        self._main_window._remember_collapsible_state(
            "detection.noise_removal", g_morph
        )

        # Morphology (Advanced/Splitting)
        g_split = CollapsibleGroupBox("Split Touching")
        self.bg_accordion.addCollapsible(g_split)
        vl_split = QVBoxLayout()
        vl_split.addWidget(
            self._main_window._create_help_label(
                "Split touching animals using body-size-aware erosion only in locally crowded regions. "
                "Enable only if animals frequently touch."
            )
        )
        f_split = QFormLayout(None)
        f_split.setHorizontalSpacing(10)
        f_split.setVerticalSpacing(8)
        self.chk_conservative_split = QCheckBox("Use conservative split")
        self.chk_conservative_split.setChecked(True)
        self.chk_conservative_split.setToolTip(
            "Locally raise the detection threshold inside suspected merged\n"
            "blobs to separate touching animals at their weakest connection\n"
            "point while preserving body shape."
        )
        f_split.addRow(self.chk_conservative_split)

        h_split = QHBoxLayout()
        self.spin_conservative_kernel = QSpinBox()
        self.spin_conservative_kernel.setRange(1, 15)
        self.spin_conservative_kernel.setSingleStep(2)
        self.spin_conservative_kernel.setValue(3)
        self.spin_conservative_kernel.setFixedHeight(30)
        self.spin_conservative_kernel.setToolTip(
            "Gaussian blur kernel applied to the local difference\n"
            "image before re-thresholding (odd number, 1-15).\n"
            "Larger = smoother split boundaries.\n"
            "1 = no smoothing. Recommended: 3-5"
        )
        self.spin_conservative_erode = QSpinBox()
        self.spin_conservative_erode.setRange(1, 10)
        self.spin_conservative_erode.setValue(1)
        self.spin_conservative_erode.setFixedHeight(30)
        self.spin_conservative_erode.setToolTip(
            "Threshold boost steps (1-10).\n"
            "Each step pulls the split threshold 25%% closer to\n"
            "nearby local peaks inside suspected merged blobs.\n"
            "Higher = more aggressive local separation.\n"
            "Recommended: 1-3"
        )
        h_split.addWidget(QLabel("Blur kernel"))
        h_split.addWidget(self.spin_conservative_kernel)
        h_split.addWidget(QLabel("Boost steps"))
        h_split.addWidget(self.spin_conservative_erode)
        f_split.addRow(h_split)

        self.chk_additional_dilation = QCheckBox("Reconnect thin parts (dilation)")
        self.chk_additional_dilation.setToolTip(
            "Use dilation to reconnect thin parts (e.g., legs, antennae).\n"
            "Recommended: Enable if animals have thin appendages.\n"
            "Disable to maintain accurate body shape."
        )
        f_split.addRow(self.chk_additional_dilation)

        h_dil = QHBoxLayout()
        self.spin_dilation_kernel_size = QSpinBox()
        self.spin_dilation_kernel_size.setRange(1, 15)
        self.spin_dilation_kernel_size.setSingleStep(2)
        self.spin_dilation_kernel_size.setValue(3)
        self.spin_dilation_kernel_size.setFixedHeight(30)
        self.spin_dilation_kernel_size.setToolTip(
            "Dilation kernel size (odd number, 1-15).\n"
            "Larger = thicker reconnection.\n"
            "Recommended: 3-5"
        )
        self.spin_dilation_iterations = QSpinBox()
        self.spin_dilation_iterations.setRange(1, 10)
        self.spin_dilation_iterations.setValue(2)
        self.spin_dilation_iterations.setFixedHeight(30)
        self.spin_dilation_iterations.setToolTip(
            "Number of dilation iterations (1-10).\n"
            "More iterations = thicker result.\n"
            "Recommended: 1-3"
        )
        h_dil.addWidget(QLabel("Kernel size"))
        h_dil.addWidget(self.spin_dilation_kernel_size)
        h_dil.addWidget(QLabel("Iterations"))
        h_dil.addWidget(self.spin_dilation_iterations)
        f_split.addRow(h_dil)
        vl_split.addLayout(f_split)
        g_split.setContentLayout(vl_split)

        l_bg.addWidget(g_split)
        self._main_window._remember_collapsible_state(
            "detection.split_touching", g_split
        )

        # --- Auto-Tune Detection Parameters button ---
        self.btn_bg_autotune = QPushButton("Auto-Tune Detection Parameters…")
        self.btn_bg_autotune.setToolTip(
            "Use Optuna to search for the best threshold, morphology,\n"
            "and conservative-split settings for your video."
        )
        self.btn_bg_autotune.clicked.connect(
            self._main_window._open_bg_parameter_helper
        )
        l_bg.addWidget(self.btn_bg_autotune)

        # --- Page 1: YOLO Params ---
        page_yolo = QWidget()
        page_yolo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        l_yolo = QVBoxLayout(page_yolo)
        l_yolo.setContentsMargins(0, 0, 0, 0)
        l_yolo.setSpacing(6)

        self.yolo_group = QGroupBox("YOLO")
        self._main_window._set_compact_section_widget(self.yolo_group)
        f_yolo = QFormLayout(self.yolo_group)
        f_yolo.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        f_yolo.setHorizontalSpacing(10)
        f_yolo.setVerticalSpacing(8)
        self.yolo_group.layout().setContentsMargins(9, 10, 9, 9)
        self.yolo_group.layout().setSpacing(8)
        self.yolo_group.layout().addWidget(
            self._main_window._create_help_label(
                "YOLO uses a trained neural network to detect animals. Choose your model file and adjust thresholds to balance recall and false positives."
            )
        )

        self.combo_yolo_obb_mode = QComboBox()
        self.combo_yolo_obb_mode.addItems(["Direct", "Sequential (Faster)"])
        self.combo_yolo_obb_mode.setFixedHeight(30)
        self.combo_yolo_obb_mode.currentIndexChanged.connect(self._on_yolo_mode_changed)
        self.combo_yolo_obb_mode.setToolTip(
            "Direct: run OBB on full frame.\n"
            "Sequential: run detect model first, crop detections, then run OBB on crops."
        )
        f_yolo.addRow("YOLO OBB mode", self.combo_yolo_obb_mode)

        self.lbl_obb_mode_warning = QLabel()
        self.lbl_obb_mode_warning.setWordWrap(True)
        self.lbl_obb_mode_warning.setStyleSheet(
            "color: #f0ad4e; font-style: italic; padding: 2px 0px;"
        )
        self.lbl_obb_mode_warning.setVisible(False)
        f_yolo.addRow("", self.lbl_obb_mode_warning)

        self.combo_yolo_model = QComboBox()
        self.combo_yolo_model.activated.connect(self.on_yolo_model_changed)
        self.combo_yolo_model.currentIndexChanged.connect(
            lambda _index: (
                self._sync_model_selector_buttons(),
                self._main_window._auto_apply_yolo_training_params("obb_direct"),
            )
        )
        self.combo_yolo_model.setFixedHeight(30)
        self.combo_yolo_model.setToolTip("Direct-mode YOLO OBB model.")
        self.btn_remove_yolo_model = self._create_model_remove_button(
            "Remove the selected direct OBB model from the local repository."
        )
        self.btn_remove_yolo_model.clicked.connect(
            lambda: self._main_window._handle_remove_selected_yolo_model(
                combo=self.combo_yolo_model,
                refresh_callback=self._refresh_yolo_model_combo,
                selection_callback=self._main_window._set_yolo_model_selection,
                model_kind="direct OBB model",
            )
        )
        self.direct_model_row_widget = self._build_model_selector_row(
            self.combo_yolo_model,
            self.btn_remove_yolo_model,
        )
        f_yolo.addRow("Direct OBB model", self.direct_model_row_widget)

        self.combo_yolo_detect_model = QComboBox()
        self.combo_yolo_detect_model.activated.connect(
            self.on_yolo_detect_model_changed
        )
        self.combo_yolo_detect_model.currentIndexChanged.connect(
            lambda _index: (
                self._sync_model_selector_buttons(),
                self._main_window._auto_apply_yolo_training_params("seq_detect"),
            )
        )
        self.combo_yolo_detect_model.setFixedHeight(30)
        self.combo_yolo_detect_model.setToolTip(
            "Sequential stage-1 model (axis-aligned detect)."
        )
        self.btn_remove_yolo_detect_model = self._create_model_remove_button(
            "Remove the selected sequential detect model from the local repository."
        )
        self.btn_remove_yolo_detect_model.clicked.connect(
            lambda: self._main_window._handle_remove_selected_yolo_model(
                combo=self.combo_yolo_detect_model,
                refresh_callback=self._refresh_yolo_detect_model_combo,
                selection_callback=self._main_window._set_yolo_detect_model_selection,
                model_kind="sequential detect model",
            )
        )
        self.seq_detect_model_row_widget = self._build_model_selector_row(
            self.combo_yolo_detect_model,
            self.btn_remove_yolo_detect_model,
        )
        f_yolo.addRow("Seq detect model", self.seq_detect_model_row_widget)

        self.combo_yolo_crop_obb_model = QComboBox()
        self.combo_yolo_crop_obb_model.activated.connect(
            self.on_yolo_crop_obb_model_changed
        )
        self.combo_yolo_crop_obb_model.currentIndexChanged.connect(
            lambda _index: (
                self._sync_model_selector_buttons(),
                self._main_window._auto_apply_yolo_training_params("seq_crop_obb"),
            )
        )
        self.combo_yolo_crop_obb_model.setFixedHeight(30)
        self.combo_yolo_crop_obb_model.setToolTip(
            "Sequential stage-2 OBB model trained on cropped detections."
        )
        self.btn_remove_yolo_crop_obb_model = self._create_model_remove_button(
            "Remove the selected sequential crop OBB model from the local repository."
        )
        self.btn_remove_yolo_crop_obb_model.clicked.connect(
            lambda: self._main_window._handle_remove_selected_yolo_model(
                combo=self.combo_yolo_crop_obb_model,
                refresh_callback=self._refresh_yolo_crop_obb_model_combo,
                selection_callback=self._main_window._set_yolo_crop_obb_model_selection,
                model_kind="sequential crop OBB model",
            )
        )
        self.seq_crop_obb_model_row_widget = self._build_model_selector_row(
            self.combo_yolo_crop_obb_model,
            self.btn_remove_yolo_crop_obb_model,
        )
        f_yolo.addRow("Seq crop OBB model", self.seq_crop_obb_model_row_widget)

        self.yolo_seq_advanced = CollapsibleGroupBox(
            "Sequential Advanced Settings", initially_expanded=False
        )
        self.yolo_seq_advanced_content = QWidget()
        f_seq_adv = QFormLayout(self.yolo_seq_advanced_content)
        self.spin_yolo_seq_crop_pad = QDoubleSpinBox()
        self.spin_yolo_seq_crop_pad.setRange(0.0, 1.0)
        self.spin_yolo_seq_crop_pad.setSingleStep(0.01)
        self.spin_yolo_seq_crop_pad.setValue(0.15)
        self.spin_yolo_seq_crop_pad.setFixedHeight(30)
        f_seq_adv.addRow("Crop pad ratio", self.spin_yolo_seq_crop_pad)
        self.spin_yolo_seq_min_crop_px = QSpinBox()
        self.spin_yolo_seq_min_crop_px.setRange(8, 1024)
        self.spin_yolo_seq_min_crop_px.setValue(64)
        self.spin_yolo_seq_min_crop_px.setFixedHeight(30)
        f_seq_adv.addRow("Min crop size (px)", self.spin_yolo_seq_min_crop_px)
        self.chk_yolo_seq_square_crop = QCheckBox("Enforce square crop")
        self.chk_yolo_seq_square_crop.setChecked(True)
        f_seq_adv.addRow("", self.chk_yolo_seq_square_crop)
        self.spin_yolo_seq_detect_conf = QDoubleSpinBox()
        self.spin_yolo_seq_detect_conf.setRange(0.01, 1.0)
        self.spin_yolo_seq_detect_conf.setSingleStep(0.01)
        self.spin_yolo_seq_detect_conf.setValue(0.25)
        self.spin_yolo_seq_detect_conf.setFixedHeight(30)
        self.spin_yolo_seq_detect_conf.setToolTip(
            "Minimum confidence for the stage-1 detection model (sequential mode only).\n"
            "Lower = more crops sent to stage-2 (higher recall, slower).\n"
            "Higher = fewer crops (faster, may miss occluded animals).\n"
            "Recommended: 0.1–0.3"
        )
        f_seq_adv.addRow("Stage-1 detect conf", self.spin_yolo_seq_detect_conf)
        self.spin_yolo_seq_stage2_imgsz = QSpinBox()
        self.spin_yolo_seq_stage2_imgsz.setRange(0, 2048)
        self.spin_yolo_seq_stage2_imgsz.setValue(160)
        self.spin_yolo_seq_stage2_imgsz.setFixedHeight(30)
        self.spin_yolo_seq_stage2_imgsz.setToolTip(
            "Crop OBB stage input size in pixels. Set 0 to disable pre-resize."
        )
        f_seq_adv.addRow("Stage-2 imgsz (px)", self.spin_yolo_seq_stage2_imgsz)
        self.spin_yolo_seq_individual_batch_size = QSpinBox()
        self.spin_yolo_seq_individual_batch_size.setRange(1, 1024)
        self.spin_yolo_seq_individual_batch_size.setValue(
            int(
                self._main_window.advanced_config.get(
                    "yolo_seq_individual_batch_size", 16
                )
            )
        )
        self.spin_yolo_seq_individual_batch_size.setFixedHeight(30)
        self.spin_yolo_seq_individual_batch_size.setToolTip(
            "Maximum number of sequential crops to send to stage-2 OBB at once.\n"
            "Non-realtime mode first batches frames, then groups crops across those frames using this size.\n"
            "Realtime mode still fixes frame batching to 1, but stage-2 crop batching uses this value."
        )
        f_seq_adv.addRow("Stage-2 crop batch", self.spin_yolo_seq_individual_batch_size)
        self.chk_yolo_seq_stage2_pow2_pad = QCheckBox(
            "Pad stage-2 batch to power-of-two"
        )
        self.chk_yolo_seq_stage2_pow2_pad.setChecked(False)
        self.chk_yolo_seq_stage2_pow2_pad.setToolTip(
            "Reduces dynamic batch-size variants in sequential stage-2 inference."
        )
        f_seq_adv.addRow("", self.chk_yolo_seq_stage2_pow2_pad)
        self.yolo_seq_advanced.setContentLayout(f_seq_adv)
        f_yolo.addRow(self.yolo_seq_advanced)

        self.spin_yolo_confidence = QDoubleSpinBox()
        self.spin_yolo_confidence.setRange(0.01, 1.0)
        self.spin_yolo_confidence.setValue(0.25)
        self.spin_yolo_confidence.setFixedHeight(30)
        self.spin_yolo_confidence.setToolTip(
            "Minimum confidence score for YOLO detections (0.01-1.0).\n"
            "Lower = more detections (more false positives).\n"
            "Higher = fewer detections (may miss animals).\n"
            "Recommended: 0.2-0.4"
        )
        self.spin_yolo_iou = QDoubleSpinBox()
        self.spin_yolo_iou.setRange(0.01, 1.0)
        self.spin_yolo_iou.setValue(0.7)
        self.spin_yolo_iou.setFixedHeight(30)
        self.spin_yolo_iou.setToolTip(
            "Intersection-over-Union threshold for non-max suppression (0.01-1.0).\n"
            "Lower = more aggressive duplicate removal.\n"
            "Higher = keep more overlapping detections.\n"
            "Recommended: 0.5-0.8"
        )
        _yolo_thresh_row = QHBoxLayout()
        _yolo_thresh_row.addWidget(QLabel("Confidence:"))
        _yolo_thresh_row.addWidget(self.spin_yolo_confidence, 1)
        _yolo_thresh_row.addSpacing(8)
        _yolo_thresh_row.addWidget(QLabel("IOU:"))
        _yolo_thresh_row.addWidget(self.spin_yolo_iou, 1)
        f_yolo.addRow(_yolo_thresh_row)

        self.chk_use_custom_obb_iou = QCheckBox("Use custom OBB overlap filtering")
        self.chk_use_custom_obb_iou.setChecked(True)
        self.chk_use_custom_obb_iou.setEnabled(False)
        self.chk_use_custom_obb_iou.setToolTip(
            "Custom polygon-based OBB IOU filtering is always enabled.\n"
            "This improves overlap handling consistency across cached and live detections."
        )
        self.chk_use_custom_obb_iou.setVisible(False)

        self.line_yolo_classes = QLineEdit()
        self.line_yolo_classes.setFixedHeight(30)
        self.line_yolo_classes.setPlaceholderText("e.g. 15, 16 (Empty for all)")
        self.line_yolo_classes.setToolTip(
            "Comma-separated class IDs to detect (leave empty for all classes).\n"
            "Example: '0,1,2' to detect only classes 0, 1, and 2.\n"
            "Refer to your model's class definitions."
        )
        f_yolo.addRow("Classes (optional)", self.line_yolo_classes)
        self._on_yolo_mode_changed(self.combo_yolo_obb_mode.currentIndex())

        l_yolo.addWidget(self.yolo_group)

        # ============================================================
        # Live Detection Batching (drives the real InferenceRunner pipeline)
        # ============================================================
        self.g_live_batching = QGroupBox("Live Detection Batching")
        self._main_window._set_compact_section_widget(self.g_live_batching)
        vl_live_batch = QVBoxLayout(self.g_live_batching)
        vl_live_batch.setSpacing(6)
        vl_live_batch.addWidget(
            self._main_window._create_help_label(
                "Controls how many frames the detector processes per GPU call during an actual "
                "tracking run. Higher batches are faster on TensorRT/CUDA/MPS; some runtimes are "
                "locked to 1 (see notice below)."
            )
        )
        self.spin_detection_batch_size = QSpinBox()
        self.spin_detection_batch_size.setRange(1, 64)
        self.spin_detection_batch_size.setValue(
            int(self._main_window.advanced_config.get("detection_batch_size", 1))
        )
        self.spin_detection_batch_size.setFixedHeight(30)
        self.spin_detection_batch_size.setToolTip(
            "Number of frames the detector processes per GPU call during live tracking.\n"
            "Feeds InferenceConfig.detection_batch_size directly.\n"
            "Higher = faster on TensorRT/CUDA/MPS, more GPU memory used.\n"
            "Typical values: 4-16 depending on GPU."
        )
        self.spin_detection_batch_size.valueChanged.connect(
            self._on_detection_batch_size_changed
        )
        _live_batch_row = QHBoxLayout()
        _live_batch_row.addWidget(QLabel("Frame batch size"))
        _live_batch_row.addWidget(self.spin_detection_batch_size, 1)
        vl_live_batch.addLayout(_live_batch_row)

        self.lbl_batch_policy_notice = QLabel("")
        self.lbl_batch_policy_notice.setWordWrap(True)
        self.lbl_batch_policy_notice.setStyleSheet(
            "color: #d7ba7d; font-size: 11px; padding-top: 2px;"
        )
        self.lbl_batch_policy_notice.setVisible(False)
        vl_live_batch.addWidget(self.lbl_batch_policy_notice)
        l_yolo.addWidget(self.g_live_batching)

        # ============================================================
        # Legacy Detector Batching (Parameter Helper cache build / Preview
        # "Test Detection" / Benchmark baseline only — NOT live tracking)
        # ============================================================
        self.g_gpu_accel = QGroupBox(
            "Legacy Detector Batching (Cache Build / Preview / Benchmark)"
        )
        self._main_window._set_compact_section_widget(self.g_gpu_accel)
        vl_gpu = QVBoxLayout(self.g_gpu_accel)
        vl_gpu.setSpacing(6)
        vl_gpu.addWidget(
            self._main_window._create_help_label(
                "These settings do not affect live tracking speed. They only apply to the "
                "legacy detector used by Parameter Helper's detection-cache build, the Preview "
                "“Test Detection” action, and the Benchmark panel's starting batch size."
            )
        )

        # TensorRT Optimization
        self.chk_enable_tensorrt = QCheckBox("TensorRT engine")
        self.chk_enable_tensorrt.setChecked(False)
        self.chk_enable_tensorrt.setEnabled(TENSORRT_AVAILABLE)

        tensorrt_tooltip = (
            "Enable NVIDIA TensorRT for the legacy detector (cache build / preview only).\n"
            "Requires NVIDIA GPU with CUDA.\n"
            "First run will export model (1-5 min), then cached for future use.\n"
            "Uses FP16 precision for maximum speed.\n"
        )
        if TENSORRT_AVAILABLE:
            tensorrt_tooltip += "\n✓ TensorRT is available on this system"
        else:
            tensorrt_tooltip += (
                "\n✗ TensorRT not available (requires NVIDIA GPU + tensorrt package)"
            )

        self.chk_enable_tensorrt.setToolTip(tensorrt_tooltip)
        self.chk_enable_tensorrt.stateChanged.connect(self._on_tensorrt_toggled)

        self.spin_tensorrt_batch = QSpinBox()
        self.spin_tensorrt_batch.setRange(1, 64)
        self.spin_tensorrt_batch.setValue(
            self._main_window.advanced_config.get("tensorrt_max_batch_size", 16)
        )
        self.spin_tensorrt_batch.setFixedHeight(30)
        self.spin_tensorrt_batch.setToolTip(
            "Maximum batch size for the legacy detector's TensorRT engine (cache build / preview only).\n"
            "Higher = potentially faster, Lower = more stable.\n"
            "Reduce if build fails (try 8, 4, or 1).\n"
            "Typical: 16-32 (high-end), 8-16 (mid-range), 1-8 (low VRAM)"
        )
        self.lbl_tensorrt_batch = QLabel("TensorRT max batch")
        self.lbl_tensorrt_batch.setStyleSheet(
            "font-size: 10px; font-weight: 600; color: #bdbdbd;"
        )

        self.chk_enable_yolo_batching = QCheckBox("GPU batching")
        self.chk_enable_yolo_batching.setChecked(
            self._main_window.advanced_config.get("enable_yolo_batching", True)
        )
        self.chk_enable_yolo_batching.setToolTip(
            "Enable batched inference in the legacy detector (cache build / preview only).\n"
            "Does not affect live tracking's detection speed."
        )
        self.chk_enable_yolo_batching.stateChanged.connect(
            self._on_yolo_batching_toggled
        )

        self.combo_yolo_batch_mode = QComboBox()
        self.combo_yolo_batch_mode.addItems(["Auto", "Manual"])
        self.combo_yolo_batch_mode.setFixedHeight(30)
        self.combo_yolo_batch_mode.setToolTip(
            "Auto: Automatically estimate the legacy detector's batch size from GPU memory.\n"
            "Manual: Specify a fixed batch size for the legacy detector."
        )
        self.combo_yolo_batch_mode.currentIndexChanged.connect(
            self._on_yolo_batch_mode_changed
        )
        self.lbl_yolo_batch_mode = QLabel("Batch mode")
        self.lbl_yolo_batch_mode.setStyleSheet(
            "font-size: 10px; font-weight: 600; color: #bdbdbd;"
        )

        self.spin_yolo_batch_size = QSpinBox()
        self.spin_yolo_batch_size.setRange(1, 64)
        self.spin_yolo_batch_size.setValue(
            self._main_window.advanced_config.get("yolo_manual_batch_size", 16)
        )
        self.spin_yolo_batch_size.setFixedHeight(30)
        self.spin_yolo_batch_size.setToolTip(
            "Manual batch size for the legacy detector (only used when mode is Manual).\n"
            "Used by Parameter Helper cache build, Preview Test Detection, and as the\n"
            "Benchmark panel's starting batch size. Does not affect live tracking.\n"
            "Typical values: 8-32 depending on GPU."
        )
        self.spin_yolo_batch_size.valueChanged.connect(
            self._on_yolo_manual_batch_size_changed
        )
        self.lbl_yolo_batch_size = QLabel("Frame batch")
        self.lbl_yolo_batch_size.setStyleSheet(
            "font-size: 10px; font-weight: 600; color: #bdbdbd;"
        )

        accel_toggle_grid = QGridLayout()
        accel_toggle_grid.setContentsMargins(0, 0, 0, 0)
        accel_toggle_grid.setHorizontalSpacing(12)
        accel_toggle_grid.setVerticalSpacing(6)
        accel_toggle_grid.addWidget(self.chk_enable_yolo_batching, 0, 0)
        accel_toggle_grid.addWidget(self.chk_enable_tensorrt, 0, 1)
        accel_toggle_grid.setColumnStretch(0, 1)
        accel_toggle_grid.setColumnStretch(1, 1)
        vl_gpu.addLayout(accel_toggle_grid)

        accel_controls_grid = QGridLayout()
        accel_controls_grid.setContentsMargins(0, 0, 0, 0)
        accel_controls_grid.setHorizontalSpacing(12)
        accel_controls_grid.setVerticalSpacing(4)
        accel_controls_grid.addWidget(self.lbl_yolo_batch_mode, 0, 0)
        accel_controls_grid.addWidget(self.lbl_yolo_batch_size, 0, 1)
        accel_controls_grid.addWidget(self.lbl_tensorrt_batch, 0, 2)
        accel_controls_grid.addWidget(self.combo_yolo_batch_mode, 1, 0)
        accel_controls_grid.addWidget(self.spin_yolo_batch_size, 1, 1)
        accel_controls_grid.addWidget(self.spin_tensorrt_batch, 1, 2)
        accel_controls_grid.setColumnStretch(0, 1)
        accel_controls_grid.setColumnStretch(1, 1)
        accel_controls_grid.setColumnStretch(2, 1)
        vl_gpu.addLayout(accel_controls_grid)
        l_yolo.addWidget(self.g_gpu_accel)

        # Set initial visibility for TensorRT widgets
        self.chk_enable_tensorrt.setVisible(False)
        self.spin_tensorrt_batch.setVisible(False)
        self.lbl_tensorrt_batch.setVisible(False)

        # Set initial visibility for batching widgets
        initial_batching_enabled = self.chk_enable_yolo_batching.isChecked()
        self.combo_yolo_batch_mode.setVisible(initial_batching_enabled)
        self.lbl_yolo_batch_mode.setVisible(initial_batching_enabled)
        self.spin_yolo_batch_size.setVisible(initial_batching_enabled)
        self.lbl_yolo_batch_size.setVisible(initial_batching_enabled)
        self.combo_yolo_batch_mode.setEnabled(initial_batching_enabled)
        self.spin_yolo_batch_size.setEnabled(False)  # Auto mode by default
        self._sync_batch_policy_controls()
        # Add pages to stack
        self.stack_detection.addWidget(page_bg)
        self.stack_detection.addWidget(page_yolo)

        vbox.addWidget(self.stack_detection)

        # ============================================================
        # Detection Overlays (moved from Visuals tab)
        # ============================================================
        # Background Subtraction specific overlays
        self.g_overlays_bg = QGroupBox("Background Diagnostics")
        self._main_window._set_compact_section_widget(self.g_overlays_bg)
        v_ov_bg = QVBoxLayout(self.g_overlays_bg)
        v_ov_bg.addWidget(
            self._main_window._create_help_label(
                "Debug background subtraction by viewing the foreground mask (detected movement) "
                "and background model (learned static image)."
            )
        )

        self.chk_show_fg = QCheckBox("Show Foreground Mask")
        self.chk_show_fg.setChecked(True)
        self.chk_show_bg = QCheckBox("Show Background Model")
        self.chk_show_bg.setChecked(True)
        _bg_ov_row = QHBoxLayout()
        _bg_ov_row.addWidget(self.chk_show_fg)
        _bg_ov_row.addWidget(self.chk_show_bg)
        v_ov_bg.addLayout(_bg_ov_row)

        vbox.addWidget(self.g_overlays_bg)

        # YOLO specific overlays
        self.g_overlays_yolo = QGroupBox("YOLO Diagnostics")
        self._main_window._set_compact_section_widget(self.g_overlays_yolo)
        v_ov_yolo = QVBoxLayout(self.g_overlays_yolo)
        v_ov_yolo.addWidget(
            self._main_window._create_help_label(
                "Show oriented bounding boxes from YOLO detection. Useful for debugging detection quality "
                "and verifying model performance."
            )
        )

        self.chk_show_yolo_obb = QCheckBox("Show YOLO OBB Detection Boxes")
        self.chk_show_yolo_obb.setChecked(False)
        v_ov_yolo.addWidget(self.chk_show_yolo_obb)

        vbox.addWidget(self.g_overlays_yolo)

        # Initially show/hide based on detection method (will be set properly by combo)
        self.g_overlays_bg.setVisible(True)
        self.g_overlays_yolo.setVisible(False)

        # ============================================================
        # Reference Scale (size + aspect ratio)
        # ============================================================
        g_ref_scale = QGroupBox("Reference Scale")
        self._main_window._set_compact_section_widget(g_ref_scale)
        vl_ref_scale = QVBoxLayout(g_ref_scale)
        vl_ref_scale.addWidget(
            self._main_window._create_help_label(
                "Define the spatial scale for tracking. These reference values make all distance, "
                "size, and shape parameters portable across videos and species. Set them BEFORE "
                "configuring tracking parameters. Use 'Test Detection' then the Auto-Set buttons "
                "to have values estimated automatically from a sample frame."
            )
        )
        fl_ref = QFormLayout(None)
        fl_ref.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        fl_ref.setHorizontalSpacing(10)
        fl_ref.setVerticalSpacing(8)

        self.spin_reference_body_size = QDoubleSpinBox()
        self.spin_reference_body_size.setRange(1.0, 500.0)
        self.spin_reference_body_size.setSingleStep(1.0)
        self.spin_reference_body_size.setValue(20.0)
        self.spin_reference_body_size.setDecimals(2)
        self.spin_reference_body_size.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.spin_reference_body_size.setFixedHeight(30)
        self.spin_reference_body_size.setToolTip(
            "Reference animal body diameter in pixels (at resize=1.0).\n"
            "All distance/size parameters are scaled relative to this value."
        )
        self.spin_reference_body_size.valueChanged.connect(self._update_body_size_info)
        fl_ref.addRow("Reference body size (px)", self.spin_reference_body_size)

        self.label_body_size_info = QLabel()
        self.label_body_size_info.setStyleSheet(
            "color: #6a6a6a; font-size: 10px; font-style: italic;"
        )
        fl_ref.addRow("", self.label_body_size_info)

        self.spin_reference_aspect_ratio = QDoubleSpinBox()
        self.spin_reference_aspect_ratio.setRange(1.0, 20.0)
        self.spin_reference_aspect_ratio.setSingleStep(0.1)
        self.spin_reference_aspect_ratio.setDecimals(2)
        self.spin_reference_aspect_ratio.setValue(2.0)
        self.spin_reference_aspect_ratio.setFixedHeight(30)
        self.spin_reference_aspect_ratio.setToolTip(
            "Species-typical major/minor axis ratio.\n"
            "Used for adaptive canonical crop dimensions and aspect ratio filtering.\n"
            "Click 'Auto-Set Aspect Ratio' to detect from sample frames."
        )
        fl_ref.addRow("Reference aspect ratio", self.spin_reference_aspect_ratio)

        vl_ref_scale.addLayout(fl_ref)

        self.label_detection_stats = QLabel(
            "No detection data yet.\nRun 'Test Detection' to estimate sizes."
        )
        self.label_detection_stats.setStyleSheet(
            "color: #9a9a9a; font-size: 11px; padding: 8px; "
            "background-color: #252526; border-radius: 4px;"
        )
        self.label_detection_stats.setWordWrap(True)
        vl_ref_scale.addWidget(self.label_detection_stats)

        btn_layout = QHBoxLayout()
        self.btn_auto_set_body_size = QPushButton("Auto-Set Body Size from Median")
        self.btn_auto_set_body_size.clicked.connect(
            self._main_window._auto_set_body_size_from_detection
        )
        self.btn_auto_set_body_size.setEnabled(False)
        self.btn_auto_set_body_size.setToolTip(
            "Automatically set reference body size to the median detected diameter"
        )
        btn_layout.addWidget(self.btn_auto_set_body_size)

        self.btn_auto_set_aspect_ratio = QPushButton("Auto-Set Aspect Ratio")
        self.btn_auto_set_aspect_ratio.clicked.connect(
            self._main_window._auto_set_aspect_ratio_from_detection
        )
        self.btn_auto_set_aspect_ratio.setEnabled(False)
        self.btn_auto_set_aspect_ratio.setToolTip(
            "Set reference aspect ratio from the median detected major/minor ratio"
        )
        btn_layout.addWidget(self.btn_auto_set_aspect_ratio)
        vl_ref_scale.addLayout(btn_layout)

        vbox.addWidget(g_ref_scale)

        # ============================================================
        # Detection Filters (size + aspect ratio ranges)
        # ============================================================
        g_filters = QGroupBox("Detection Filters")
        self._main_window._set_compact_section_widget(g_filters)
        vl_filters = QVBoxLayout(g_filters)
        vl_filters.addWidget(
            self._main_window._create_help_label(
                "Filter detections by size and aspect ratio relative to the reference values above. "
                "Enabling these removes noise, debris, and erroneous clusters before tracking."
            )
        )
        f_filters = QFormLayout(None)
        f_filters.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        f_filters.setHorizontalSpacing(10)
        f_filters.setVerticalSpacing(8)

        self.chk_size_filtering = QCheckBox("Filter detections by size")
        self.chk_size_filtering.setToolTip(
            "Filter detected objects by area to remove noise and artifacts.\n"
            "Recommended: Enable for cleaner tracking."
        )
        f_filters.addRow(self.chk_size_filtering)

        h_sf = QHBoxLayout()
        self.spin_min_object_size = QDoubleSpinBox()
        self.spin_min_object_size.setRange(0.1, 5.0)
        self.spin_min_object_size.setSingleStep(0.1)
        self.spin_min_object_size.setDecimals(2)
        self.spin_min_object_size.setValue(0.3)
        self.spin_min_object_size.setFixedHeight(30)
        self.spin_min_object_size.setToolTip(
            "Minimum object area as multiple of reference body area.\n"
            "Filters out small noise/artifacts.\n"
            "Recommended: 0.2-0.5× (allows partial occlusion)"
        )
        self.spin_max_object_size = QDoubleSpinBox()
        self.spin_max_object_size.setRange(0.5, 10.0)
        self.spin_max_object_size.setSingleStep(0.1)
        self.spin_max_object_size.setDecimals(2)
        self.spin_max_object_size.setValue(3.0)
        self.spin_max_object_size.setFixedHeight(30)
        self.spin_max_object_size.setToolTip(
            "Maximum object area as multiple of reference body area.\n"
            "Filters out large clusters or artifacts.\n"
            "Recommended: 2-4× (handles overlapping animals)"
        )
        h_sf.addWidget(QLabel("Min size (body lengths)"))
        h_sf.addWidget(self.spin_min_object_size)
        h_sf.addWidget(QLabel("Max size (body lengths)"))
        h_sf.addWidget(self.spin_max_object_size)
        f_filters.addRow(h_sf)

        self.chk_enable_aspect_ratio_filtering = QCheckBox(
            "Filter detections by aspect ratio"
        )
        self.chk_enable_aspect_ratio_filtering.setChecked(False)
        self.chk_enable_aspect_ratio_filtering.setToolTip(
            "Reject detections with aspect ratios outside the expected range.\n"
            "Helps filter scratches, debris, and other non-animal detections."
        )
        f_filters.addRow(self.chk_enable_aspect_ratio_filtering)

        h_ar_mult = QHBoxLayout()
        self.spin_min_ar_multiplier = QDoubleSpinBox()
        self.spin_min_ar_multiplier.setRange(0.1, 1.0)
        self.spin_min_ar_multiplier.setSingleStep(0.05)
        self.spin_min_ar_multiplier.setDecimals(2)
        self.spin_min_ar_multiplier.setValue(0.5)
        self.spin_min_ar_multiplier.setFixedHeight(30)
        self.spin_min_ar_multiplier.setToolTip(
            "Minimum aspect ratio = reference × this multiplier.\n"
            "Detections more compact than this are rejected."
        )
        self.spin_max_ar_multiplier = QDoubleSpinBox()
        self.spin_max_ar_multiplier.setRange(1.0, 10.0)
        self.spin_max_ar_multiplier.setSingleStep(0.1)
        self.spin_max_ar_multiplier.setDecimals(2)
        self.spin_max_ar_multiplier.setValue(2.0)
        self.spin_max_ar_multiplier.setFixedHeight(30)
        self.spin_max_ar_multiplier.setToolTip(
            "Maximum aspect ratio = reference × this multiplier.\n"
            "Detections more elongated than this are rejected."
        )
        h_ar_mult.addWidget(QLabel("Min multiplier"))
        h_ar_mult.addWidget(self.spin_min_ar_multiplier)
        h_ar_mult.addWidget(QLabel("Max multiplier"))
        h_ar_mult.addWidget(self.spin_max_ar_multiplier)
        f_filters.addRow(h_ar_mult)

        vl_filters.addLayout(f_filters)
        vbox.addWidget(g_filters)

        scroll.setWidget(content)
        layout.addWidget(scroll)
        self._sync_model_selector_buttons()

    def apply_config(self, config: TrackerConfig) -> None:
        """Update panel widgets to reflect a new config object."""
        self._config = config

    # =========================================================================
    # QUERY HELPERS (moved from MainWindow)
    # =========================================================================

    def _is_yolo_detection_mode(self) -> bool:
        """Return True when current detection mode is YOLO OBB."""
        return self.combo_detection_method.currentIndex() == 1

    def _is_identity_analysis_enabled(self) -> bool:
        """Return effective runtime state for identity classification."""
        if not hasattr(self._main_window, "_identity_panel"):
            return False
        return bool(
            self._main_window._identity_panel.g_identity.isChecked()
            and self._is_yolo_detection_mode()
        )

    def _selected_identity_method(self) -> str:
        """Return canonical identity-method key for runtime/config usage."""
        if not self._is_identity_analysis_enabled():
            return "none_disabled"
        cfg = self._identity_config()
        has_apriltags = cfg.get("use_apriltags", False)
        has_cnn = bool(cfg.get("cnn_classifiers", []))
        if has_apriltags and not has_cnn:
            return "apriltags"
        if has_cnn and not has_apriltags:
            return "cnn_classifier"
        if has_apriltags or has_cnn:
            return "cnn_classifier"  # multi-method: report as cnn_classifier for compat
        return "none_disabled"

    def _build_identity_config(self, *, require_enabled_gate: bool) -> dict:
        """Collect AprilTag/CNN identity config from the identity panel."""
        if require_enabled_gate and not self._is_identity_analysis_enabled():
            return {"use_apriltags": False, "cnn_classifiers": []}
        if not self._is_yolo_detection_mode():
            return {"use_apriltags": False, "cnn_classifiers": []}
        ip = getattr(self._main_window, "_identity_panel", None)
        use_apriltags = ip is not None and ip.g_apriltags.isChecked()
        cnn_classifiers = []
        if ip is not None:
            for row in ip._cnn_classifier_rows():
                cfg = row.to_config()
                if cfg is not None:
                    cnn_classifiers.append(cfg)
        return {
            "use_apriltags": use_apriltags,
            "cnn_classifiers": cnn_classifiers,
        }

    def _identity_config(self) -> dict:
        """Return use_apriltags + cnn_classifiers config dict."""
        return self._build_identity_config(require_enabled_gate=True)

    def _preview_identity_config(self) -> dict:
        """Return preview-only identity overlays without requiring the master gate."""
        return self._build_identity_config(require_enabled_gate=False)

    # =========================================================================
    # YOLO BATCHING / TENSORRT HANDLERS (moved from MainWindow)
    # =========================================================================

    def _on_yolo_batching_toggled(self, state):
        """Enable/disable YOLO batching controls based on checkbox."""
        self._sync_batch_policy_controls()

    def _on_yolo_manual_batch_size_changed(self, value: int):
        """Keep legacy fixed-batch field synchronized for fixed runtimes."""
        if self._main_window._runtime_requires_fixed_yolo_batch() and hasattr(
            self, "spin_tensorrt_batch"
        ):
            self.spin_tensorrt_batch.setValue(int(value))
        self._sync_batch_policy_controls()

    def _on_yolo_batch_mode_changed(self, index):
        """Show/hide manual batch size based on selected mode."""
        self._sync_batch_policy_controls()

    def _on_tensorrt_toggled(self, state):
        """Enable/disable TensorRT batch size control based on checkbox."""
        if not self.chk_enable_tensorrt.isVisible():
            self.spin_tensorrt_batch.setVisible(False)
            self.lbl_tensorrt_batch.setVisible(False)
            return
        self._sync_batch_policy_controls()

    def _on_detection_batch_size_changed(self, value: int):
        """Refresh the live batching notice when the frame batch size changes."""
        self._sync_live_detection_batch_controls()

    def _sync_batch_policy_controls(self) -> None:
        """Keep legacy-detector batching controls aligned with runtime policy.

        These widgets (chk_enable_yolo_batching, combo_yolo_batch_mode,
        spin_yolo_batch_size, chk_enable_tensorrt, spin_tensorrt_batch) only
        affect the legacy detector used by Parameter Helper's cache build,
        Preview "Test Detection", and the Benchmark panel's baseline — not
        live tracking. See _sync_live_detection_batch_controls for the
        control that actually drives InferenceConfig.detection_batch_size.
        """
        realtime_enabled = False
        if hasattr(self._main_window, "_setup_panel"):
            realtime_enabled = is_realtime_workflow(
                self._main_window._setup_panel.chk_realtime_mode.isChecked(),
                getattr(
                    self._main_window, "_workflow_mode_key", lambda: "non_realtime"
                )(),
            )
        fixed_runtime = self._main_window._runtime_requires_fixed_yolo_batch()
        if not hasattr(self, "chk_enable_yolo_batching"):
            return

        if fixed_runtime and self.combo_yolo_batch_mode.currentIndex() != 1:
            self.combo_yolo_batch_mode.blockSignals(True)
            self.combo_yolo_batch_mode.setCurrentIndex(1)
            self.combo_yolo_batch_mode.blockSignals(False)
        if fixed_runtime and not self.chk_enable_yolo_batching.isChecked():
            self.chk_enable_yolo_batching.blockSignals(True)
            self.chk_enable_yolo_batching.setChecked(True)
            self.chk_enable_yolo_batching.blockSignals(False)

        batching_enabled = self.chk_enable_yolo_batching.isChecked() or fixed_runtime
        manual_mode = self.combo_yolo_batch_mode.currentIndex() == 1
        tensorrt_enabled = (
            self.chk_enable_tensorrt.isVisible()
            and self.chk_enable_tensorrt.isChecked()
        )

        self.combo_yolo_batch_mode.setVisible(batching_enabled)
        self.lbl_yolo_batch_mode.setVisible(batching_enabled)
        self.spin_yolo_batch_size.setVisible(batching_enabled)
        self.lbl_yolo_batch_size.setVisible(batching_enabled)
        self.spin_tensorrt_batch.setVisible(tensorrt_enabled)
        self.lbl_tensorrt_batch.setVisible(tensorrt_enabled)

        if realtime_enabled:
            self.chk_enable_yolo_batching.setEnabled(False)
            self.combo_yolo_batch_mode.setEnabled(False)
            self.spin_yolo_batch_size.setEnabled(False)
            self.spin_tensorrt_batch.setEnabled(False)
            self.lbl_tensorrt_batch.setEnabled(tensorrt_enabled)
        else:
            self.chk_enable_yolo_batching.setEnabled(not fixed_runtime)
            self.combo_yolo_batch_mode.setEnabled(
                batching_enabled and not fixed_runtime
            )
            self.spin_yolo_batch_size.setEnabled(
                batching_enabled and (manual_mode or fixed_runtime)
            )
            self.spin_tensorrt_batch.setEnabled(tensorrt_enabled)
            self.lbl_tensorrt_batch.setEnabled(tensorrt_enabled)

        self._sync_live_detection_batch_controls()

    def _sync_live_detection_batch_controls(self) -> None:
        """Keep the Frame batch size control aligned with runtime policy.

        This is the control that actually reaches InferenceConfig.detection_batch_size
        for live tracking runs.
        """
        if not hasattr(self, "spin_detection_batch_size"):
            return

        realtime_enabled = False
        if hasattr(self._main_window, "_setup_panel"):
            realtime_enabled = is_realtime_workflow(
                self._main_window._setup_panel.chk_realtime_mode.isChecked(),
                getattr(
                    self._main_window, "_workflow_mode_key", lambda: "non_realtime"
                )(),
            )
        sequential = self.combo_yolo_obb_mode.currentIndex() == 1
        coreml_locked = self._main_window._gpu_fast_obb_is_coreml_only()

        recommendation = self._main_window._current_detection_benchmark_recommendation()
        recommendation_text = ""
        if recommendation is not None:
            individual_batch_size = getattr(
                recommendation, "individual_batch_size", None
            )
            if individual_batch_size:
                recommendation_text = (
                    "Benchmark recommendation: "
                    f"{recommendation.runtime_label} at frame batch {recommendation.batch_size} / crop batch {int(individual_batch_size)}."
                )
            else:
                recommendation_text = f"Benchmark recommendation: {recommendation.runtime_label} at batch {recommendation.batch_size}."

        if realtime_enabled:
            self.spin_detection_batch_size.blockSignals(True)
            self.spin_detection_batch_size.setValue(1)
            self.spin_detection_batch_size.blockSignals(False)
            self.spin_detection_batch_size.setEnabled(False)
            if sequential:
                message = "Realtime tracking fixes the frame batch to 1. Sequential stage-2 crop batching still uses the Stage-2 crop batch setting."
            else:
                message = "Realtime tracking processes detection one frame at a time; frame batch size is fixed to 1."
            self.lbl_batch_policy_notice.setText(message)
            self.lbl_batch_policy_notice.setVisible(True)
            return

        if coreml_locked:
            self.spin_detection_batch_size.blockSignals(True)
            self.spin_detection_batch_size.setValue(1)
            self.spin_detection_batch_size.blockSignals(False)
            self.spin_detection_batch_size.setEnabled(False)
            message = (
                "On this platform, gpu_fast detection (OBB) runs on "
                "CoreML, which supports only batch size 1 — one frame "
                "at a time, regardless of this setting. CoreML "
                "classification (identity/head-tail/CNN) is unaffected "
                "and still batches normally."
            )
            if recommendation_text:
                message += "\n" + recommendation_text
            self.lbl_batch_policy_notice.setText(message)
            self.lbl_batch_policy_notice.setVisible(True)
            return

        self.spin_detection_batch_size.setEnabled(True)
        if sequential:
            message = (
                "Sequential mode's stage-1 detection batching showed higher "
                "run-to-run variation in detections during testing (see "
                "docs/superpowers/specs/2026-07-03-tensorrt-coreml-cross-frame-"
                "batching-design.md). Stage-2 crop batch is usually the safer "
                "place to batch."
            )
            if recommendation_text:
                message += "\n" + recommendation_text
            self.lbl_batch_policy_notice.setText(message)
            self.lbl_batch_policy_notice.setVisible(True)
        elif recommendation_text:
            self.lbl_batch_policy_notice.setText(recommendation_text)
            self.lbl_batch_policy_notice.setVisible(True)
        else:
            self.lbl_batch_policy_notice.clear()
            self.lbl_batch_policy_notice.setVisible(False)

    # =========================================================================
    # DETECTION METHOD CHANGED UI (moved from MainWindow)
    # =========================================================================

    def _on_detection_method_changed_ui(self, index):
        """Update stack widget when detection method changes."""
        self.stack_detection.setCurrentIndex(index)
        is_background_subtraction = index == 0
        self.g_img.setVisible(is_background_subtraction)
        self.g_overlays_bg.setVisible(is_background_subtraction)
        self.g_overlays_yolo.setVisible(not is_background_subtraction)
        self._update_preview_display()
        self.on_detection_method_changed(index)
        self._main_window._on_runtime_context_changed()
        self._main_window._queue_ui_state_save()

    # =========================================================================
    # IMAGE ADJUSTMENT HANDLERS (moved from MainWindow)
    # =========================================================================

    def _on_brightness_changed(self, value):
        """Handle brightness slider change."""
        self.label_brightness_val.setText(str(value))
        self._main_window.detection_test_result = None
        self._update_preview_display()

    def _on_contrast_changed(self, value):
        """Handle contrast slider change."""
        contrast_val = value / 100.0
        self.label_contrast_val.setText(f"{contrast_val:.2f}")
        self._main_window.detection_test_result = None
        self._update_preview_display()

    def _on_gamma_changed(self, value):
        """Handle gamma slider change."""
        gamma_val = value / 100.0
        self.label_gamma_val.setText(f"{gamma_val:.2f}")
        self._main_window.detection_test_result = None
        self._update_preview_display()

    def _on_zoom_changed(self, value):
        """Handle zoom slider change."""
        zoom_val = value / 100.0
        self._main_window.label_zoom_val.setText(f"{zoom_val:.2f}x")

        # If tracking is active, re-render from the last received frame so that
        # zoom does not flash a stale detection-test or preview image.
        tracking_worker = getattr(self._main_window, "tracking_worker", None)
        if tracking_worker is not None and tracking_worker.isRunning():
            last_rgb = getattr(self._main_window, "_last_tracking_frame_rgb", None)
            if last_rgb is not None:
                from PySide6.QtCore import Qt
                from PySide6.QtGui import QImage, QPixmap

                z = max(value / 100.0, 0.1)
                h, w, _ = last_rgb.shape
                qimg = QImage(last_rgb.data, w, h, w * 3, QImage.Format_RGB888)
                scaled = qimg.scaled(
                    int(w * z), int(h * z), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self._main_window._set_video_pixmap(QPixmap.fromImage(scaled))
            return

        if self._main_window.detection_test_result is not None:
            self._redisplay_detection_test()
        elif getattr(self._main_window, "roi_base_frame", None) is not None and getattr(
            self._main_window, "roi_shapes", None
        ):
            self._main_window._display_roi_with_zoom()
        else:
            self._update_preview_display()

    # =========================================================================
    # BODY SIZE INFO (moved from MainWindow)
    # =========================================================================

    def _update_body_size_info(self):
        """Update the info label showing calculated body area."""
        body_size = self.spin_reference_body_size.value()
        body_area = math.pi * (body_size / 2.0) ** 2
        self.label_body_size_info.setText(
            f"\u2248 {body_area:.1f} px\u00b2 area (all size/distance params scale with this)"
        )

    # =========================================================================
    # DETECTION STATISTICS (moved from MainWindow)
    # =========================================================================

    def _update_detection_stats(self, detected_dimensions, resize_factor=1.0):
        """Update detection statistics display."""
        if not detected_dimensions or len(detected_dimensions) == 0:
            self.label_detection_stats.setText(
                "No detections found.\nAdjust parameters and try again."
            )
            self.btn_auto_set_body_size.setEnabled(False)
            self._main_window.detected_sizes = None
            return

        scale_factor = 1.0 / resize_factor
        major_axes = [dims[0] * scale_factor for dims in detected_dimensions]
        minor_axes = [dims[1] * scale_factor for dims in detected_dimensions]

        aspect_ratios = [
            major / minor if minor > 0 else 1.0
            for major, minor in zip(major_axes, minor_axes)
        ]
        geometric_means = [
            math.sqrt(major * minor) for major, minor in zip(major_axes, minor_axes)
        ]

        stats = {
            "major": {
                "mean": np.mean(major_axes),
                "median": np.median(major_axes),
                "std": np.std(major_axes),
                "min": np.min(major_axes),
                "max": np.max(major_axes),
            },
            "minor": {
                "mean": np.mean(minor_axes),
                "median": np.median(minor_axes),
                "std": np.std(minor_axes),
                "min": np.min(minor_axes),
                "max": np.max(minor_axes),
            },
            "aspect_ratio": {
                "mean": np.mean(aspect_ratios),
                "median": np.median(aspect_ratios),
                "std": np.std(aspect_ratios),
            },
            "geometric_mean": {
                "mean": np.mean(geometric_means),
                "median": np.median(geometric_means),
                "std": np.std(geometric_means),
            },
        }

        self._main_window.detected_sizes = {
            "major_axes": major_axes,
            "minor_axes": minor_axes,
            "aspect_ratios": aspect_ratios,
            "geometric_means": geometric_means,
            "stats": stats,
            "count": len(detected_dimensions),
            "resize_factor": resize_factor,
            "recommended_body_size": stats["geometric_mean"]["median"],
            "recommended_aspect_ratio": stats["aspect_ratio"]["median"],
        }

        stats_text = (
            f"Analyzed {len(detected_dimensions)} detections:\n\n"
            f"Major Axis (length):\n"
            f"  \u2022 Median: {stats['major']['median']:.1f} px  (range: {stats['major']['min']:.1f} - {stats['major']['max']:.1f})\n"
            f"  \u2022 Mean: {stats['major']['mean']:.1f} \u00b1 {stats['major']['std']:.1f} px\n\n"
            f"Minor Axis (width):\n"
            f"  \u2022 Median: {stats['minor']['median']:.1f} px  (range: {stats['minor']['min']:.1f} - {stats['minor']['max']:.1f})\n"
            f"  \u2022 Mean: {stats['minor']['mean']:.1f} \u00b1 {stats['minor']['std']:.1f} px\n\n"
            f"Aspect Ratio (length/width):\n"
            f"  \u2022 Median: {stats['aspect_ratio']['median']:.2f}  Mean: {stats['aspect_ratio']['mean']:.2f} \u00b1 {stats['aspect_ratio']['std']:.2f}\n\n"
            f"Recommended Body Size: {stats['geometric_mean']['median']:.1f} px\n"
            f"  (geometric mean of dimensions)"
        )
        self.label_detection_stats.setText(stats_text)
        self.btn_auto_set_body_size.setEnabled(True)
        self.btn_auto_set_aspect_ratio.setEnabled(True)

    # =========================================================================
    # PREVIEW DISPLAY (moved from MainWindow)
    # =========================================================================

    def _update_preview_display(self):
        """Update the video display with current brightness/contrast/gamma settings."""
        if self._main_window.preview_frame_original is None:
            return
        if self._main_window.detection_test_result is not None:
            self._redisplay_detection_test()
            return

        brightness = self.slider_brightness.value()
        contrast = self.slider_contrast.value() / 100.0
        gamma = self.slider_gamma.value() / 100.0
        detection_method = self.combo_detection_method.currentText()
        is_background_subtraction = detection_method == "Background Subtraction"

        from hydra_suite.utils.image_processing import apply_image_adjustments

        if is_background_subtraction:
            gray = cv2.cvtColor(
                self._main_window.preview_frame_original, cv2.COLOR_RGB2GRAY
            )
            adjusted = apply_image_adjustments(
                gray, brightness, contrast, gamma, use_gpu=False
            )
            adjusted_rgb = cv2.cvtColor(adjusted, cv2.COLOR_GRAY2RGB)
        else:
            adjusted_rgb = self._main_window.preview_frame_original

        h, w, ch = adjusted_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(adjusted_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        if self._main_window.roi_mask is not None:
            qimg = self._main_window._apply_roi_mask_to_image(qimg)

        zoom_val = max(self._main_window.slider_zoom.value() / 100.0, 0.1)
        if zoom_val != 1.0:
            scaled_w = int(w * zoom_val)
            scaled_h = int(h * zoom_val)
            qimg = qimg.scaled(
                scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.FastTransformation
            )

        pixmap = QPixmap.fromImage(qimg)
        self._main_window._set_video_pixmap(pixmap)

    def _redisplay_detection_test(self):
        """Redisplay the stored detection test result with current zoom."""
        if self._main_window.detection_test_result is None:
            return

        test_frame_rgb, resize_f = self._main_window.detection_test_result
        h, w, ch = test_frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(test_frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        zoom_val = max(self._main_window.slider_zoom.value() / 100.0, 0.1)
        effective_scale = zoom_val * resize_f
        if effective_scale != 1.0:
            orig_h, orig_w = self._main_window.preview_frame_original.shape[:2]
            scaled_w = int(orig_w * effective_scale)
            scaled_h = int(orig_h * effective_scale)
            qimg = qimg.scaled(
                scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.FastTransformation
            )

        pixmap = QPixmap.fromImage(qimg)
        self._main_window._set_video_pixmap(pixmap)

    # =========================================================================
    # PREVIEW DETECTION TEST (moved from MainWindow)
    # =========================================================================

    def _test_detection_on_preview(self):
        """Test detection algorithm on the current preview frame."""
        from hydra_suite.trackerkit.gui.workers.preview_worker import (
            PreviewDetectionWorker,
        )

        if self._main_window.preview_frame_original is None:
            logger.warning("No preview frame loaded")
            return

        if (
            self._main_window.preview_detection_worker
            and self._main_window.preview_detection_worker.isRunning()
        ):
            logger.info("Preview detection is already running")
            return

        use_detection_filters = False
        detection_filters_enabled = bool(
            self.chk_size_filtering.isChecked()
            or self.chk_enable_aspect_ratio_filtering.isChecked()
        )
        if detection_filters_enabled:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("Detection Filter Options")
            msg.setText("Detection filters are currently enabled!")
            msg.setInformativeText(
                "For accurate size estimation, it's recommended to run detection\n"
                "WITHOUT detection constraints. However, you can test with constraints\n"
                "if you want to see how filtering affects the results.\n\n"
                "This includes both size and aspect-ratio filtering.\n\n"
                "How would you like to proceed?"
            )
            btn_without = msg.addButton(
                "NO Detection Filtering (Recommended)", QMessageBox.AcceptRole
            )
            btn_with = msg.addButton("WITH Detection Filtering", QMessageBox.ActionRole)
            btn_cancel = msg.addButton("Cancel", QMessageBox.RejectRole)
            msg.setDefaultButton(btn_without)
            msg.exec()
            clicked = msg.clickedButton()
            if clicked == btn_cancel:
                return
            elif clicked == btn_with:
                use_detection_filters = True
                logger.info("Running detection test WITH detection filtering enabled")
            else:
                use_detection_filters = False
                logger.info(
                    "Running detection test WITHOUT detection filtering (recommended for size estimation)"
                )

        context = self._collect_preview_detection_context()
        if (
            int(context.get("detection_method", 0)) == 1
            and str(context.get("yolo_obb_mode", "direct")).strip().lower()
            == "sequential"
        ):
            detect_model = str(context.get("yolo_detect_model_path", "")).strip()
            crop_obb_model = str(context.get("yolo_crop_obb_model_path", "")).strip()
            if not detect_model or not crop_obb_model:
                QMessageBox.warning(
                    self,
                    "Missing Sequential Models",
                    "Sequential YOLO OBB mode in detection preview requires both a detect model and a crop OBB model.",
                )
                return
        self._main_window.preview_detection_worker = PreviewDetectionWorker(
            self._main_window.preview_frame_original.copy(),
            context,
            use_detection_filters,
        )
        self._main_window.preview_detection_worker.finished_signal.connect(
            self._on_preview_detection_finished
        )
        self._main_window.preview_detection_worker.error_signal.connect(
            self._on_preview_detection_error
        )
        self._main_window.preview_detection_worker.finished.connect(
            self._on_preview_detection_worker_finished
        )
        self._main_window._set_preview_test_running(True)
        self._main_window.preview_detection_worker.start()

    def _collect_preview_detection_context(self) -> dict:
        """Capture current UI values for async preview detection."""
        from hydra_suite.runtime.compute_runtime import derive_pose_runtime_settings
        from hydra_suite.runtime.resolver import (
            detect_platform,
            resolve_compute_runtime,
        )

        selected_runtime = self._main_window._preview_safe_runtime(
            self._main_window._selected_compute_runtime()
        )
        tier = self._main_window._selected_runtime_tier()
        platform = detect_platform()
        obb_compute_runtime = resolve_compute_runtime(tier, platform, stage="obb")
        identity_cfg = self._preview_identity_config()
        ip = getattr(self._main_window, "_identity_panel", None)
        pose_backend_family = (
            ip.combo_pose_model_type.currentText().strip().lower()
            if ip is not None
            else "yolo"
        )
        runtime_pose = derive_pose_runtime_settings(
            selected_runtime, backend_family=pose_backend_family
        )
        class_text = self.line_yolo_classes.text().strip()
        target_classes = None
        if class_text:
            try:
                target_classes = [int(x.strip()) for x in class_text.split(",")]
            except ValueError:
                target_classes = None

        sp = getattr(self._main_window, "_setup_panel", None)
        return {
            "detection_method": self.combo_detection_method.currentIndex(),
            "video_path": sp.file_line.text() if sp is not None else "",
            "bg_prime_seconds": self.spin_bg_prime.value(),
            "fps": sp.spin_fps.value() if sp is not None else 25.0,
            "brightness": self.slider_brightness.value(),
            "contrast": self.slider_contrast.value() / 100.0,
            "gamma": self.slider_gamma.value() / 100.0,
            "roi_mask": (
                self._main_window.roi_mask.copy()
                if self._main_window.roi_mask is not None
                else None
            ),
            "resize_factor": sp.spin_resize.value() if sp is not None else 1.0,
            "dark_on_light": self.chk_dark_on_light.isChecked(),
            "threshold_value": self.spin_threshold.value(),
            "morph_kernel_size": self.spin_morph_size.value(),
            "enable_additional_dilation": self.chk_additional_dilation.isChecked(),
            "dilation_kernel_size": self.spin_dilation_kernel_size.value(),
            "dilation_iterations": self.spin_dilation_iterations.value(),
            "min_contour": self.spin_min_contour.value(),
            "reference_body_size": self.spin_reference_body_size.value(),
            "reference_aspect_ratio": self.spin_reference_aspect_ratio.value(),
            "enable_aspect_ratio_filtering": self.chk_enable_aspect_ratio_filtering.isChecked(),
            "min_aspect_ratio_multiplier": self.spin_min_ar_multiplier.value(),
            "max_aspect_ratio_multiplier": self.spin_max_ar_multiplier.value(),
            "min_object_size": self.spin_min_object_size.value(),
            "max_object_size": self.spin_max_object_size.value(),
            "compute_runtime": selected_runtime,
            "obb_compute_runtime": obb_compute_runtime,
            "headtail_runtime": (
                self._main_window._selected_headtail_runtime()
                if hasattr(self._main_window, "_selected_headtail_runtime")
                else selected_runtime
            ),
            "cnn_runtime": (
                self._main_window._selected_cnn_runtime()
                if hasattr(self._main_window, "_selected_cnn_runtime")
                else selected_runtime
            ),
            "yolo_obb_mode": (
                "sequential"
                if self.combo_yolo_obb_mode.currentIndex() == 1
                else "direct"
            ),
            "yolo_model_path": self._main_window._get_selected_yolo_model_path(),
            "yolo_obb_direct_model_path": self._main_window._get_selected_yolo_model_path(),
            "yolo_detect_model_path": self._main_window._get_selected_yolo_detect_model_path(),
            "yolo_crop_obb_model_path": self._main_window._get_selected_yolo_crop_obb_model_path(),
            "headtail_enabled": (
                ip.g_headtail.isChecked() if ip is not None else False
            ),
            "configured_headtail_model_path": (
                ip._get_configured_yolo_headtail_model_path() if ip is not None else ""
            ),
            "yolo_headtail_model_path": (
                ip._get_selected_yolo_headtail_model_path() if ip is not None else ""
            ),
            "pose_overrides_headtail": (
                ip.chk_pose_overrides_headtail.isChecked() if ip is not None else False
            ),
            "headtail_batch_size": (
                ip.spin_headtail_batch.value() if ip is not None else 64
            ),
            "yolo_seq_crop_pad_ratio": self.spin_yolo_seq_crop_pad.value(),
            "yolo_seq_min_crop_size_px": self.spin_yolo_seq_min_crop_px.value(),
            "yolo_seq_enforce_square_crop": self.chk_yolo_seq_square_crop.isChecked(),
            "yolo_seq_stage2_imgsz": self.spin_yolo_seq_stage2_imgsz.value(),
            "yolo_seq_individual_batch_size": self.spin_yolo_seq_individual_batch_size.value(),
            "yolo_seq_stage2_pow2_pad": self.chk_yolo_seq_stage2_pow2_pad.isChecked(),
            "yolo_seq_detect_conf_threshold": self.spin_yolo_seq_detect_conf.value(),
            "yolo_headtail_conf_threshold": (
                ip.spin_yolo_headtail_conf.value() if ip is not None else 0.25
            ),
            "yolo_headtail_detect_conf_threshold": (
                ip.spin_yolo_headtail_detect_conf.value() if ip is not None else 0.25
            ),
            "yolo_confidence": self.spin_yolo_confidence.value(),
            "yolo_iou": self.spin_yolo_iou.value(),
            "yolo_target_classes": target_classes,
            "max_targets": sp.spin_max_targets.value() if sp is not None else 10,
            "max_contour_multiplier": self.spin_max_contour_multiplier.value(),
            "enable_conservative_split": self.chk_conservative_split.isChecked(),
            "conservative_kernel_size": self.spin_conservative_kernel.value(),
            "conservative_erode_iterations": self.spin_conservative_erode.value(),
            "use_apriltags": identity_cfg.get("use_apriltags", False),
            "cnn_classifiers": identity_cfg.get("cnn_classifiers", []),
            "apriltag_family": (
                ip.combo_apriltag_family.currentText() if ip is not None else "tag36h11"
            ),
            "apriltag_decimate": (
                ip.spin_apriltag_decimate.value() if ip is not None else 1.0
            ),
            "enable_pose_extractor": self._main_window._is_pose_inference_enabled(),
            "pose_model_type": pose_backend_family,
            "pose_model_dir": self._main_window._get_resolved_pose_model_dir(
                pose_backend_family
            ),
            "pose_runtime_flavor": runtime_pose["pose_runtime_flavor"],
            "pose_min_kpt_conf_valid": (
                ip.spin_pose_min_kpt_conf_valid.value() if ip is not None else 0.5
            ),
            "pose_skeleton_file": (
                ip.line_pose_skeleton_file.text().strip() if ip is not None else ""
            ),
            "pose_ignore_keypoints": self._main_window._parse_pose_ignore_keypoints(),
            "pose_direction_anterior_keypoints": self._main_window._parse_pose_direction_anterior_keypoints(),
            "pose_direction_posterior_keypoints": self._main_window._parse_pose_direction_posterior_keypoints(),
            "pose_batch_size": ip.spin_pose_batch.value() if ip is not None else 1,
            "pose_sleap_env": self._main_window._selected_pose_sleap_env(),
            "pose_sleap_device": runtime_pose["pose_sleap_device"],
            "individual_crop_padding": (
                ip.spin_individual_padding.value() if ip is not None else 0.1
            ),
            "individual_background_color": (
                [int(c) for c in ip._background_color] if ip is not None else [0, 0, 0]
            ),
            "suppress_foreign_obb_regions": (
                ip.chk_suppress_foreign_obb.isChecked() if ip is not None else False
            ),
        }

    @Slot(dict)
    def _on_preview_detection_finished(self, result: dict):
        """Handle successful async preview detection completion."""
        test_frame_rgb = result.get("test_frame_rgb")
        resize_f = float(result.get("resize_factor", 1.0))
        detected_dimensions = result.get("detected_dimensions") or []
        if test_frame_rgb is None:
            logger.warning("Preview detection completed without image result")
            return
        self._update_detection_stats(detected_dimensions, resize_f)
        self._main_window.detection_test_result = (test_frame_rgb.copy(), resize_f)
        h, w, ch = test_frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(test_frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        zoom_val = max(self._main_window.slider_zoom.value() / 100.0, 0.1)
        effective_scale = zoom_val * resize_f
        if (
            effective_scale != 1.0
            and self._main_window.preview_frame_original is not None
        ):
            orig_h, orig_w = self._main_window.preview_frame_original.shape[:2]
            scaled_w = int(orig_w * effective_scale)
            scaled_h = int(orig_h * effective_scale)
            qimg = qimg.scaled(
                scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        self._main_window._set_video_pixmap(QPixmap.fromImage(qimg))
        self._main_window._fit_image_to_screen()
        logger.info("Detection test completed on preview frame")

    @Slot(str)
    def _on_preview_detection_error(self, error_message: str):
        """Handle async preview detection failure."""
        logger.error(f"Detection test failed: {error_message}")
        QMessageBox.warning(
            self,
            "Detection Test Failed",
            "Detection test failed on preview frame. Check logs for details.",
        )

    @Slot()
    def _on_preview_detection_worker_finished(self):
        """Finalize async preview detection UI state and worker lifecycle."""
        sender = self.sender()
        if sender is self._main_window.preview_detection_worker:
            try:
                sender.deleteLater()
            except Exception:
                pass
            self._main_window.preview_detection_worker = None
        self._main_window._set_preview_test_running(False)

    # =========================================================================
    # DETECTION METHOD CHANGED (moved from MainWindow)
    # =========================================================================

    def on_detection_method_changed(self, index: object) -> object:
        """Keep compatibility hook and synchronize YOLO-only individual-analysis controls."""
        self._main_window._sync_individual_analysis_mode_ui()

    # =========================================================================
    # YOLO MODEL COMBO REFRESH (moved from MainWindow)
    # =========================================================================

    def _refresh_yolo_model_combo(self, preferred_model_path: object = None) -> object:
        """Populate direct OBB model combo from repository models."""
        self._main_window._populate_yolo_model_combo(
            self.combo_yolo_model,
            preferred_model_path=preferred_model_path,
            default_path="",
            include_none=False,
            task_family="obb",
            usage_role="obb_direct",
        )
        self._sync_model_selector_buttons()

    def _refresh_yolo_detect_model_combo(self, preferred_model_path: object = None):
        self._main_window._populate_yolo_model_combo(
            self.combo_yolo_detect_model,
            preferred_model_path=preferred_model_path,
            default_path="",
            include_none=True,
            task_family="detect",
            usage_role="seq_detect",
        )
        self._sync_model_selector_buttons()

    def _refresh_yolo_crop_obb_model_combo(self, preferred_model_path: object = None):
        self._main_window._populate_yolo_model_combo(
            self.combo_yolo_crop_obb_model,
            preferred_model_path=preferred_model_path,
            default_path="",
            include_none=True,
            task_family="obb",
            usage_role="seq_crop_obb",
        )
        self._sync_model_selector_buttons()

    @staticmethod
    def _create_model_remove_button(tooltip: str) -> QPushButton:
        """Create a compact remove button for a model-selector row."""
        button = QPushButton("-")
        button.setObjectName("SecondaryBtn")
        button.setFixedSize(28, 30)
        button.setToolTip(tooltip)
        return button

    @staticmethod
    def _build_model_selector_row(
        combo: QComboBox, remove_button: QPushButton
    ) -> QWidget:
        """Return a combo row with a dedicated remove button."""
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(combo, 1)
        row.addWidget(remove_button, 0)
        return widget

    @staticmethod
    def _combo_has_selected_model(combo: QComboBox) -> bool:
        """Return True when the combo currently points to a removable model."""
        selected_data = combo.currentData(Qt.UserRole)
        return bool(selected_data and selected_data not in ("__add_new__", "__none__"))

    def _sync_model_selector_buttons(self) -> None:
        """Enable remove buttons only when their combos point to real models."""
        button_pairs = (
            (self.combo_yolo_model, self.btn_remove_yolo_model),
            (self.combo_yolo_detect_model, self.btn_remove_yolo_detect_model),
            (self.combo_yolo_crop_obb_model, self.btn_remove_yolo_crop_obb_model),
        )
        for combo, button in button_pairs:
            button.setEnabled(self._combo_has_selected_model(combo))

    # =========================================================================
    # YOLO MODE CHANGED (moved from MainWindow)
    # =========================================================================

    def _on_yolo_mode_changed(self, _index: object) -> object:
        """Toggle direct/sequential model controls."""
        form = self.yolo_group.layout()

        def _set_row_visible(widget: object, visible: bool):
            if widget is None:
                return
            widget.setVisible(bool(visible))
            if form is None:
                return
            try:
                label = form.labelForField(widget)
            except Exception:
                label = None
            if label is not None:
                label.setVisible(bool(visible))

        sequential = self.combo_yolo_obb_mode.currentIndex() == 1
        _set_row_visible(getattr(self, "combo_yolo_model", None), not sequential)
        _set_row_visible(getattr(self, "combo_yolo_detect_model", None), sequential)
        _set_row_visible(getattr(self, "combo_yolo_crop_obb_model", None), sequential)
        _set_row_visible(getattr(self, "yolo_seq_advanced", None), sequential)

        ip = getattr(self._main_window, "_identity_panel", None)
        _set_row_visible(
            getattr(self._main_window, "headtail_model_row_widget", None), True
        )
        _set_row_visible(
            getattr(self._main_window, "chk_pose_overrides_headtail", None), True
        )
        if ip is not None:
            ip.spin_yolo_headtail_conf.setEnabled(
                bool(ip._get_selected_yolo_headtail_model_path().strip())
            )
        self._main_window._update_obb_mode_warning()
        self._sync_batch_policy_controls()

    # =========================================================================
    # YOLO MODEL CHANGED (moved from MainWindow)
    # =========================================================================

    def on_yolo_model_changed(self, index: object) -> object:
        """Handle direct OBB model selection."""
        if self.combo_yolo_model.itemData(index, Qt.UserRole) == "__add_new__":
            self._main_window._handle_add_new_yolo_model(
                combo=self.combo_yolo_model,
                refresh_callback=self._refresh_yolo_model_combo,
                selection_callback=self._main_window._set_yolo_model_selection,
                task_family="obb",
                usage_role="obb_direct",
                dialog_title="Add Direct OBB Model",
            )
            return
        self._on_yolo_mode_changed(index)

    def on_yolo_detect_model_changed(self, index: object) -> object:
        """Handle sequential detection model combo-box changes, opening the add-model dialog when the sentinel item is selected."""
        if self.combo_yolo_detect_model.itemData(index, Qt.UserRole) == "__add_new__":
            self._main_window._handle_add_new_yolo_model(
                combo=self.combo_yolo_detect_model,
                refresh_callback=self._refresh_yolo_detect_model_combo,
                selection_callback=self._main_window._set_yolo_detect_model_selection,
                task_family="detect",
                usage_role="seq_detect",
                dialog_title="Add Sequential Detect Model",
            )
            return
        self._on_yolo_mode_changed(index)

    def on_yolo_crop_obb_model_changed(self, index: object) -> object:
        """Handle sequential crop OBB model combo-box changes, opening the add-model dialog when the sentinel item is selected."""
        if self.combo_yolo_crop_obb_model.itemData(index, Qt.UserRole) == "__add_new__":
            self._main_window._handle_add_new_yolo_model(
                combo=self.combo_yolo_crop_obb_model,
                refresh_callback=self._refresh_yolo_crop_obb_model_combo,
                selection_callback=self._main_window._set_yolo_crop_obb_model_selection,
                task_family="obb",
                usage_role="seq_crop_obb",
                dialog_title="Add Sequential Crop OBB Model",
            )
            return
        self._on_yolo_mode_changed(index)
        self._main_window._apply_crop_obb_training_params()
