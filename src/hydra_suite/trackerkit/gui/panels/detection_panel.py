"""DetectionPanel — detection method, image preprocessing, and model config."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, Slot
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
from hydra_suite.utils.gpu_utils import MPS_AVAILABLE, TORCH_CUDA_AVAILABLE
from hydra_suite.widgets.workers import BaseWorker

if TYPE_CHECKING:
    from hydra_suite.trackerkit.gui.main_window import MainWindow

logger = logging.getLogger(__name__)

#: Direct-model checkpoint tasks -> combo indices (combo_yolo_direct_task is the
#: hidden serialized state holder; the visible label is auto-inferred).
_DIRECT_TASK_INDEX = {"obb": 0, "detect": 1, "segment": 2}
_DIRECT_TASK_LABELS = {
    "obb": "OBB (native)",
    "detect": "Detect (fixed angle)",
    "segment": "Segment (rotated mask)",
}


#: In-flight task-inference workers. Holds a strong reference until each worker
#: finishes, so closing the window mid-inference can never destroy a running
#: QThread (abort). Entries are removed by the worker's own `finished` handler.
_INFLIGHT_TASK_WORKERS: set = set()


class _DirectTaskInferenceWorker(BaseWorker):
    """Background checkpoint-properties reader.

    Loading an ultralytics checkpoint takes seconds on large OBB models, so the
    task/imgsz are read off the GUI thread. Emits ``props_inferred`` with
    ``(task, imgsz)`` — task is 'obb'|'detect'|'segment'|'pose'|'classify'|''
    and imgsz is the trained input size (0 when unknown) — and lets the panel
    decide whether the result is still current.
    """

    props_inferred: Signal = Signal(str, int)

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self._model_path = str(model_path or "")
        _INFLIGHT_TASK_WORKERS.add(self)
        self.finished.connect(self._release_inflight_ref)

    def _release_inflight_ref(self) -> None:
        _INFLIGHT_TASK_WORKERS.discard(self)

    def execute(self) -> None:
        from hydra_suite.core.inference.model_paths import (
            infer_checkpoint_imgsz,
            infer_checkpoint_task,
        )

        self.props_inferred.emit(
            infer_checkpoint_task(self._model_path),
            infer_checkpoint_imgsz(self._model_path),
        )


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
        self._task_worker: BaseWorker | None = None
        self._seq_crop_worker: BaseWorker | None = None
        self._task_kick_scheduled = False
        self._seq_crop_kick_scheduled = False
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
        f_yolo = QGridLayout(self.yolo_group)
        f_yolo.setContentsMargins(9, 10, 9, 9)
        f_yolo.setHorizontalSpacing(14)
        f_yolo.setVerticalSpacing(6)
        f_yolo.setColumnStretch(0, 1)
        f_yolo.setColumnStretch(1, 1)
        f_yolo.addWidget(
            self._main_window._create_help_label(
                "YOLO uses a trained neural network to detect animals. Choose your model file and adjust thresholds to balance recall and false positives."
            ),
            0,
            0,
            1,
            2,
        )

        def _yolo_label(text: str, tooltip: str = "") -> QLabel:
            lbl = QLabel(text)
            if tooltip:
                lbl.setToolTip(tooltip)
            return lbl

        def _labeled_row(
            label_text: str, field: QWidget, *, tooltip: str = ""
        ) -> QWidget:
            row = QWidget()
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(6)
            lay.addWidget(_yolo_label(label_text, tooltip))
            lay.addWidget(field, 1)
            return row

        self.combo_yolo_obb_mode = QComboBox()
        self.combo_yolo_obb_mode.addItems(["Direct", "Sequential (Faster)"])
        self.combo_yolo_obb_mode.setFixedHeight(30)
        self.combo_yolo_obb_mode.currentIndexChanged.connect(self._on_yolo_mode_changed)
        self.combo_yolo_obb_mode.setToolTip(
            "Direct: run OBB on full frame.\n"
            "Sequential: run detect model first, crop detections, then run OBB on crops."
        )
        self.row_mode = _labeled_row(
            "YOLO OBB mode",
            self.combo_yolo_obb_mode,
            tooltip=(
                "Direct: run OBB on full frame.\n"
                "Sequential: run detect model first, crop detections, then run OBB on crops."
            ),
        )
        f_yolo.addWidget(self.row_mode, 1, 0, 1, 2)

        self.lbl_obb_mode_warning = QLabel()
        self.lbl_obb_mode_warning.setWordWrap(True)
        self.lbl_obb_mode_warning.setStyleSheet(
            "color: #f0ad4e; font-style: italic; padding: 2px 0px;"
        )
        self.lbl_obb_mode_warning.setVisible(False)
        f_yolo.addWidget(self.lbl_obb_mode_warning, 2, 0, 1, 2)

        # ------------------------------------------------------------------
        # Direct model selector + inferred task (left column of the grid).
        # ------------------------------------------------------------------
        self.combo_yolo_model = QComboBox()
        self.combo_yolo_model.activated.connect(self.on_yolo_model_changed)
        self.combo_yolo_model.currentIndexChanged.connect(
            lambda _index: (
                self._sync_model_selector_buttons(),
                self._main_window._auto_apply_yolo_training_params("obb_direct"),
                self._kick_direct_task_inference(),
            )
        )
        self.combo_yolo_model.setFixedHeight(30)
        self.combo_yolo_model.setToolTip(
            "Direct detection model (native OBB, plain detect, or segmentation checkpoint)."
        )
        self.btn_remove_yolo_model = self._create_model_remove_button(
            "Remove the selected direct model from the local repository."
        )
        self.btn_remove_yolo_model.clicked.connect(
            lambda: self._main_window._handle_remove_selected_yolo_model(
                combo=self.combo_yolo_model,
                refresh_callback=self._refresh_yolo_model_combo,
                selection_callback=self._main_window._set_yolo_model_selection,
                model_kind="direct model",
            )
        )
        self.direct_model_row_widget = self._build_model_selector_row(
            self.combo_yolo_model,
            self.btn_remove_yolo_model,
        )

        # Inferred direct-model task. The combo below is kept ONLY as the
        # serialized state holder (config save/load, dataset-panel export
        # levels) and is driven automatically from the checkpoint; users never
        # see or edit it. The read-only label is the visible affordance.
        self.combo_yolo_direct_task = QComboBox()
        self.combo_yolo_direct_task.addItems(
            ["OBB (native)", "Detect (fixed angle)", "Segment (rotated mask)"]
        )
        self.combo_yolo_direct_task.setVisible(False)
        self.combo_yolo_direct_task.currentIndexChanged.connect(
            self._on_yolo_direct_task_changed
        )

        self.spin_yolo_fixed_angle = QDoubleSpinBox()
        self.spin_yolo_fixed_angle.setRange(-180.0, 180.0)
        self.spin_yolo_fixed_angle.setDecimals(1)
        self.spin_yolo_fixed_angle.setSuffix(" deg")
        self.spin_yolo_fixed_angle.setFixedHeight(30)
        self.spin_yolo_fixed_angle.setToolTip(
            "Fixed OBB angle applied to every detection when the direct model "
            "is a plain 'Detect (fixed angle)' checkpoint."
        )
        self.lbl_yolo_fixed_angle = _yolo_label(
            "Fixed angle",
            "Fixed OBB angle applied to every detection when the direct model "
            "is a plain 'Detect (fixed angle)' checkpoint.",
        )
        self.lbl_direct_task_inferred = QLabel()
        self.lbl_direct_task_inferred.setToolTip(
            "Inferred automatically from the selected checkpoint. "
            "Native OBB checkpoints detect rotated boxes directly; plain detect "
            "checkpoints get a fixed angle; segmentation checkpoints derive the "
            "angle from the predicted mask."
        )
        # Full-width row: the model selector with its auto-inferred task inline
        # ("Direct Model", since the checkpoint may be OBB, plain detect, or
        # segmentation — not necessarily OBB).
        self.row_direct_model = QWidget()
        _row_direct_lay = QHBoxLayout(self.row_direct_model)
        _row_direct_lay.setContentsMargins(0, 0, 0, 0)
        _row_direct_lay.setSpacing(6)
        _row_direct_lay.addWidget(_yolo_label("Direct Model"))
        _row_direct_lay.addWidget(self.direct_model_row_widget, 1)
        _row_direct_lay.addWidget(_yolo_label("Task"))
        _row_direct_lay.addWidget(self.lbl_direct_task_inferred)
        _row_direct_lay.addWidget(self.lbl_yolo_fixed_angle)
        _row_direct_lay.addWidget(self.spin_yolo_fixed_angle)
        f_yolo.addWidget(self.row_direct_model, 3, 0, 1, 2)

        # ------------------------------------------------------------------
        # Sliced inference (SAHI) — right column, shown only in direct mode.
        # ------------------------------------------------------------------
        self.chk_slice_enabled = QCheckBox("Enable sliced inference (SAHI)")
        self.chk_slice_enabled.setToolTip(
            "Tile each frame and detect per tile to recover small-object recall "
            "and reduce crowding. Off by default; direct mode only."
        )
        self.chk_slice_enabled.toggled.connect(self._on_slice_toggled)
        self.row_slice_toggle = _labeled_row("Sliced inference", self.chk_slice_enabled)
        f_yolo.addWidget(self.row_slice_toggle, 8, 0)

        self.combo_slice_geometry = QComboBox()
        self.combo_slice_geometry.addItems(["auto_model", "auto_object", "custom"])
        self.combo_slice_geometry.setFixedHeight(30)
        self.combo_slice_geometry.setToolTip(
            "auto_model: tile = model input size (fastest, no resample). "
            "auto_object: size tiles from expected object size. "
            "custom: explicit tile size."
        )
        self.combo_slice_geometry.currentIndexChanged.connect(
            self._on_slice_geometry_changed
        )
        self.row_slice_geometry = _labeled_row(
            "Slice geometry", self.combo_slice_geometry
        )
        f_yolo.addWidget(self.row_slice_geometry, 8, 1)

        # SAHI parameters. `custom` needs an explicit tile size, `auto_object`
        # a target object fraction; tile overlap applies to every mode. These
        # bind straight to the advanced_config keys engine_params.py reads
        # (SLICE_OVERLAP / SLICE_HEIGHT / SLICE_WIDTH / SLICE_OBJECT_TILE_FRACTION).
        advanced = self._main_window.advanced_config
        self.spin_slice_overlap = QDoubleSpinBox()
        self.spin_slice_overlap.setRange(0.0, 0.9)
        self.spin_slice_overlap.setSingleStep(0.05)
        self.spin_slice_overlap.setValue(float(advanced.get("slice_overlap", 0.2)))
        self.spin_slice_overlap.setToolTip(
            "Fraction of each tile that overlaps its neighbours (0.0–0.9). "
            "Higher overlap reduces missed detections on tile seams but repeats "
            "inference on more area."
        )
        self.spin_slice_tile_w = QSpinBox()
        self.spin_slice_tile_w.setRange(0, 8192)
        self.spin_slice_tile_w.setValue(int(advanced.get("slice_width", 0)))
        self.spin_slice_tile_w.setToolTip(
            "Custom tile width in original-frame pixels (0 = model input size)."
        )
        self.spin_slice_tile_h = QSpinBox()
        self.spin_slice_tile_h.setRange(0, 8192)
        self.spin_slice_tile_h.setValue(int(advanced.get("slice_height", 0)))
        self.spin_slice_tile_h.setToolTip(
            "Custom tile height in original-frame pixels (0 = model input size)."
        )
        self.spin_slice_object_fraction = QDoubleSpinBox()
        self.spin_slice_object_fraction.setRange(0.01, 0.9)
        self.spin_slice_object_fraction.setSingleStep(0.01)
        self.spin_slice_object_fraction.setValue(
            float(advanced.get("slice_object_tile_fraction", 0.15))
        )
        self.spin_slice_object_fraction.setToolTip(
            "Tile size for auto_object: the reference object spans this "
            "fraction of the tile."
        )
        self.lbl_slice_overlap = _yolo_label("Tile overlap")
        self.lbl_slice_tile_w = _yolo_label("Tile W (px)")
        self.lbl_slice_tile_h = _yolo_label("Tile H (px)")
        self.lbl_slice_object_fraction = _yolo_label("Object tile fraction")
        self.row_slice_params = QWidget()
        _slice_params_lay = QHBoxLayout(self.row_slice_params)
        _slice_params_lay.setContentsMargins(0, 0, 0, 0)
        _slice_params_lay.setSpacing(6)
        _slice_params_lay.addWidget(self.lbl_slice_overlap)
        _slice_params_lay.addWidget(self.spin_slice_overlap)
        _slice_params_lay.addSpacing(10)
        _slice_params_lay.addWidget(self.lbl_slice_tile_w)
        _slice_params_lay.addWidget(self.spin_slice_tile_w)
        _slice_params_lay.addWidget(self.lbl_slice_tile_h)
        _slice_params_lay.addWidget(self.spin_slice_tile_h)
        _slice_params_lay.addWidget(self.lbl_slice_object_fraction)
        _slice_params_lay.addWidget(self.spin_slice_object_fraction)
        _slice_params_lay.addStretch(1)
        f_yolo.addWidget(self.row_slice_params, 9, 0, 1, 2)

        for key, spin in (
            ("slice_overlap", self.spin_slice_overlap),
            ("slice_width", self.spin_slice_tile_w),
            ("slice_height", self.spin_slice_tile_h),
            ("slice_object_tile_fraction", self.spin_slice_object_fraction),
        ):

            def _sync_advanced(value, _key=key, _spin=spin):
                self._main_window.advanced_config[_key] = (
                    int(value) if isinstance(_spin, QSpinBox) else float(value)
                )

            spin.valueChanged.connect(_sync_advanced)

        # ------------------------------------------------------------------
        # Sequential model selectors (right column in direct mode swaps to the
        # seq selectors when the mode is Sequential).
        # ------------------------------------------------------------------
        self.combo_yolo_detect_model = QComboBox()
        self.combo_yolo_detect_model.activated.connect(
            self.on_yolo_detect_model_changed
        )
        self.combo_yolo_detect_model.currentIndexChanged.connect(
            lambda _index: (
                self._sync_model_selector_buttons(),
                self._main_window._auto_apply_yolo_training_params("seq_detect"),
                self._sync_seq_advanced_derived_state(),
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
        self.row_seq_detect = _labeled_row(
            "Seq detect model", self.seq_detect_model_row_widget
        )
        f_yolo.addWidget(self.row_seq_detect, 4, 0)

        self.combo_yolo_crop_obb_model = QComboBox()
        self.combo_yolo_crop_obb_model.activated.connect(
            self.on_yolo_crop_obb_model_changed
        )
        self.combo_yolo_crop_obb_model.currentIndexChanged.connect(
            lambda _index: (
                self._sync_model_selector_buttons(),
                self._main_window._auto_apply_yolo_training_params("seq_crop_obb"),
                self._kick_seq_crop_model_props(),
                self._sync_seq_advanced_derived_state(),
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
        self.row_seq_crop = _labeled_row(
            "Seq crop OBB model", self.seq_crop_obb_model_row_widget
        )
        f_yolo.addWidget(self.row_seq_crop, 4, 1)

        self.yolo_seq_advanced = CollapsibleGroupBox(
            "Sequential Advanced Settings", initially_expanded=False
        )
        self.yolo_seq_advanced_content = QWidget()
        # Compact 2-column grid: label|field | label|field per row.
        f_seq_adv = QGridLayout(self.yolo_seq_advanced_content)
        f_seq_adv.setHorizontalSpacing(12)
        f_seq_adv.setVerticalSpacing(6)
        f_seq_adv.setColumnStretch(1, 1)
        f_seq_adv.setColumnStretch(3, 1)

        self.spin_yolo_seq_crop_pad = QDoubleSpinBox()
        self.spin_yolo_seq_crop_pad.setRange(0.0, 1.0)
        self.spin_yolo_seq_crop_pad.setSingleStep(0.01)
        self.spin_yolo_seq_crop_pad.setValue(0.15)
        self.spin_yolo_seq_crop_pad.setFixedHeight(30)
        self.spin_yolo_seq_crop_pad.setToolTip(
            "Padding added around each stage-1 detection before cropping. "
            "Auto-set from the sequential models' training when available."
        )
        f_seq_adv.addWidget(_yolo_label("Crop pad ratio"), 0, 0)
        f_seq_adv.addWidget(self.spin_yolo_seq_crop_pad, 0, 1)

        self.spin_yolo_seq_min_crop_px = QSpinBox()
        self.spin_yolo_seq_min_crop_px.setRange(8, 1024)
        self.spin_yolo_seq_min_crop_px.setValue(64)
        self.spin_yolo_seq_min_crop_px.setFixedHeight(30)
        self.spin_yolo_seq_min_crop_px.setToolTip(
            "Smallest crop (px) sent to stage-2. "
            "Auto-set from the sequential models' training when available."
        )
        f_seq_adv.addWidget(_yolo_label("Min crop size (px)"), 0, 2)
        f_seq_adv.addWidget(self.spin_yolo_seq_min_crop_px, 0, 3)

        self.chk_yolo_seq_square_crop = QCheckBox("Enforce square crop")
        self.chk_yolo_seq_square_crop.setChecked(True)
        self.chk_yolo_seq_square_crop.setToolTip(
            "Force square stage-2 crops. "
            "Auto-set from the sequential models' training when available."
        )
        f_seq_adv.addWidget(self.chk_yolo_seq_square_crop, 1, 0, 1, 2)

        self.spin_yolo_seq_detect_conf = QDoubleSpinBox()
        self.spin_yolo_seq_detect_conf.setRange(0.01, 1.0)
        self.spin_yolo_seq_detect_conf.setSingleStep(0.01)
        self.spin_yolo_seq_detect_conf.setValue(0.25)
        self.spin_yolo_seq_detect_conf.setFixedHeight(30)
        self.spin_yolo_seq_detect_conf.setToolTip(
            "Minimum confidence for the stage-1 detection model (sequential mode only).\n"
            "A runtime choice — no model metadata records it.\n"
            "Lower = more crops sent to stage-2 (higher recall, slower).\n"
            "Higher = fewer crops (faster, may miss occluded animals).\n"
            "Recommended: 0.1–0.3"
        )
        f_seq_adv.addWidget(_yolo_label("Stage-1 detect conf"), 1, 2)
        f_seq_adv.addWidget(self.spin_yolo_seq_detect_conf, 1, 3)

        self.spin_yolo_seq_stage2_imgsz = QSpinBox()
        self.spin_yolo_seq_stage2_imgsz.setRange(0, 2048)
        self.spin_yolo_seq_stage2_imgsz.setValue(160)
        self.spin_yolo_seq_stage2_imgsz.setFixedHeight(30)
        self.spin_yolo_seq_stage2_imgsz.setToolTip(
            "Crop OBB stage input size in pixels (0 = disable pre-resize). "
            "Auto-set from the crop model's training / checkpoint when available."
        )
        f_seq_adv.addWidget(_yolo_label("Stage-2 imgsz (px)"), 2, 0)
        f_seq_adv.addWidget(self.spin_yolo_seq_stage2_imgsz, 2, 1)

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
            "A runtime/GPU choice — no model metadata records it.\n"
            "Non-realtime mode first batches frames, then groups crops across those frames using this size.\n"
            "Realtime mode still fixes frame batching to 1, but stage-2 crop batching uses this value."
        )
        f_seq_adv.addWidget(_yolo_label("Stage-2 crop batch"), 2, 2)
        f_seq_adv.addWidget(self.spin_yolo_seq_individual_batch_size, 2, 3)

        self.chk_yolo_seq_stage2_pow2_pad = QCheckBox(
            "Pad stage-2 batch to power-of-two"
        )
        self.chk_yolo_seq_stage2_pow2_pad.setChecked(False)
        self.chk_yolo_seq_stage2_pow2_pad.setToolTip(
            "Runtime choice — no model metadata records it. "
            "Reduces dynamic batch-size variants in sequential stage-2 inference."
        )
        f_seq_adv.addWidget(self.chk_yolo_seq_stage2_pow2_pad, 3, 0, 1, 2)
        self.yolo_seq_advanced.setContentLayout(f_seq_adv)
        f_yolo.addWidget(self.yolo_seq_advanced, 5, 0, 1, 2)

        # ------------------------------------------------------------------
        # Thresholds + classes — shared by both modes.
        # ------------------------------------------------------------------
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
        self.row_thresholds = QWidget()
        _yolo_thresh_row = QHBoxLayout(self.row_thresholds)
        _yolo_thresh_row.setContentsMargins(0, 0, 0, 0)
        _yolo_thresh_row.addWidget(QLabel("Confidence:"))
        _yolo_thresh_row.addWidget(self.spin_yolo_confidence, 1)
        _yolo_thresh_row.addSpacing(12)
        _yolo_thresh_row.addWidget(QLabel("IOU:"))
        _yolo_thresh_row.addWidget(self.spin_yolo_iou, 1)
        f_yolo.addWidget(self.row_thresholds, 6, 0, 1, 2)

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
        self.row_classes = _labeled_row("Classes (optional)", self.line_yolo_classes)
        f_yolo.addWidget(self.row_classes, 7, 0, 1, 2)

        self._on_yolo_mode_changed(self.combo_yolo_obb_mode.currentIndex())
        self._on_yolo_direct_task_changed(self.combo_yolo_direct_task.currentIndex())

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

        self._sync_live_detection_batch_controls()
        # Add pages to stack
        self.stack_detection.addWidget(page_bg)
        self.stack_detection.addWidget(page_yolo)

        vbox.addWidget(self.stack_detection)

        # Detection overlay diagnostics (foreground/background mask, YOLO OBB
        # boxes) moved from per-checkbox toggles into the single global Debug
        # Mode; see main_window.btn_debug_mode / _gui_display_overlay.

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
            "Sets the canonical crop canvas shape (a poor match costs background\n"
            "pixels, not accuracy).\n"
            "ALSO centres the detection aspect-ratio filter when that filter is\n"
            "enabled — changing this with filtering on changes which detections\n"
            "survive.\n"
            "Click 'Auto-Set Aspect Ratio' to measure it from sample frames."
        )
        fl_ref.addRow("Reference aspect ratio", self.spin_reference_aspect_ratio)

        self.spin_canonical_margin = QDoubleSpinBox()
        self.spin_canonical_margin.setRange(1.0, 3.0)
        self.spin_canonical_margin.setSingleStep(0.05)
        self.spin_canonical_margin.setDecimals(2)
        self.spin_canonical_margin.setValue(1.3)
        self.spin_canonical_margin.setFixedHeight(30)
        self.spin_canonical_margin.setToolTip(
            "Canonical crop canvas margin over the reference major axis.\n"
            "This is the operator's dial for avoiding clipped animals: raise it\n"
            "if large/fast-moving animals get cut off in canonical crops, at the\n"
            "cost of more background pixels per crop.\n"
            "Click 'Auto-Set Margin from Max' to size it from the largest\n"
            "detected animal in the sample frames."
        )
        fl_ref.addRow("Canonical margin", self.spin_canonical_margin)

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

        self.btn_auto_set_margin = QPushButton("Auto-Set Margin from Max")
        self.btn_auto_set_margin.clicked.connect(
            self._main_window._auto_set_margin_from_detection
        )
        self.btn_auto_set_margin.setEnabled(False)
        self.btn_auto_set_margin.setToolTip(
            "Set canonical margin so the largest detected animal's major axis\n"
            "fits inside the canonical crop canvas"
        )
        btn_layout.addWidget(self.btn_auto_set_margin)
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

    def _on_detection_batch_size_changed(self, value: int):
        """Refresh the live batching notice when the frame batch size changes."""
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
            self.lbl_batch_policy_notice.setText(message)
            self.lbl_batch_policy_notice.setVisible(True)
            return

        self.spin_detection_batch_size.setEnabled(True)
        if sequential:
            message = (
                "Sequential mode's stage-1 detection batching showed higher "
                "run-to-run variation in detections during testing (see "
                "docs/superpowers/specs/done/2026-07-03-tensorrt-coreml-cross-frame-"
                "batching-design.md). Stage-2 crop batch is usually the safer "
                "place to batch."
            )
            self.lbl_batch_policy_notice.setText(message)
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
            self.btn_auto_set_margin.setEnabled(False)
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
        self.btn_auto_set_margin.setEnabled(True)

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
        # Capture the authoritative tracking params (the SAME dict the real
        # tracking pass builds its InferenceConfig from) on the main thread,
        # while Qt widgets are safe to read. The preview YOLO branch uses this
        # as its config source so the preview runs the EXACT detection config
        # the full run will -- most importantly the SLICE_* (SAHI) keys, which
        # the preview's own param mapping never carried, so it silently ran
        # non-sliced while the run sliced (spurious detections diverged).
        try:
            context["tracking_params"] = self._main_window.get_parameters_dict()
        except Exception:
            logger.warning(
                "Could not capture tracking params for preview; falling back "
                "to preview-local param mapping.",
                exc_info=True,
            )
            context["tracking_params"] = None
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
        tier = self._main_window._selected_runtime_tier()
        identity_cfg = self._preview_identity_config()
        ip = getattr(self._main_window, "_identity_panel", None)
        pose_backend_family = (
            ip.combo_pose_model_type.currentText().strip().lower()
            if ip is not None
            else "yolo"
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
            "runtime_tier": tier,
            "yolo_obb_mode": (
                "sequential"
                if self.combo_yolo_obb_mode.currentIndex() == 1
                else "direct"
            ),
            "yolo_obb_direct_task": ["obb", "detect", "segment"][
                self.combo_yolo_direct_task.currentIndex()
            ],
            "yolo_fixed_angle_deg": self.spin_yolo_fixed_angle.value(),
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
        # The initial selection is made silently while signals are blocked
        # (QComboBox auto-selects the first item on addItem), so the
        # currentIndexChanged hook never fires for it — kick the checkpoint
        # task inference explicitly here to cover init / config-load / import.
        self._kick_direct_task_inference()

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
        # Same silent-first-selection note as _refresh_yolo_model_combo: the
        # currentIndexChanged hook misses the initial pick, so apply the model's
        # training defaults explicitly here.
        self._main_window._auto_apply_yolo_training_params("seq_detect")
        self._sync_seq_advanced_derived_state()

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
        # Silent-first-selection gap: apply training defaults + kick the
        # checkpoint-imgsz fallback explicitly.
        self._main_window._auto_apply_yolo_training_params("seq_crop_obb")
        self._kick_seq_crop_model_props()
        self._sync_seq_advanced_derived_state()

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

    @staticmethod
    def _set_widget_visible(widget: object, visible: bool) -> None:
        """Set a widget's visibility without touching any form-label bookkeeping."""
        if widget is not None:
            widget.setVisible(bool(visible))

    def _on_yolo_mode_changed(self, _index: object) -> object:
        """Toggle direct/sequential model controls.

        Only the controls relevant to the selected mode are shown: sequential
        selectors / advanced settings appear solely in Sequential mode, and the
        direct model / inferred task / SAHI controls solely in Direct mode.
        Within Direct mode the SAHI geometry row additionally depends on the
        SAHI checkbox (see ``_on_slice_toggled``).
        """
        sequential = self.combo_yolo_obb_mode.currentIndex() == 1

        # Direct-mode controls (left column of the YOLO grid).
        self._set_widget_visible(
            getattr(self, "row_direct_model", None), not sequential
        )
        self._set_widget_visible(
            getattr(self, "row_slice_toggle", None), not sequential
        )
        self._set_widget_visible(
            getattr(self, "row_slice_geometry", None),
            not sequential and self.chk_slice_enabled.isChecked(),
        )
        self._set_widget_visible(
            getattr(self, "row_slice_params", None),
            not sequential and self.chk_slice_enabled.isChecked(),
        )
        if not sequential and self.chk_slice_enabled.isChecked():
            self._on_slice_geometry_changed(self.combo_slice_geometry.currentIndex())

        # Sequential-mode controls (right column of the YOLO grid).
        self._set_widget_visible(getattr(self, "row_seq_detect", None), sequential)
        self._set_widget_visible(getattr(self, "row_seq_crop", None), sequential)
        self._set_widget_visible(getattr(self, "yolo_seq_advanced", None), sequential)

        if sequential:
            self._set_widget_visible(
                getattr(self, "spin_yolo_fixed_angle", None), False
            )
            self._set_widget_visible(getattr(self, "lbl_yolo_fixed_angle", None), False)
        else:
            self._on_yolo_direct_task_changed(None)

        ip = getattr(self._main_window, "_identity_panel", None)
        self._set_widget_visible(
            getattr(self._main_window, "headtail_model_row_widget", None), True
        )
        self._set_widget_visible(
            getattr(self._main_window, "chk_pose_overrides_headtail", None), True
        )
        if ip is not None:
            ip.spin_yolo_headtail_conf.setEnabled(
                bool(ip._get_selected_yolo_headtail_model_path().strip())
            )
        self._main_window._update_obb_mode_warning()
        self._sync_live_detection_batch_controls()
        if hasattr(self._main_window, "_dataset_panel"):
            self._main_window._dataset_panel.refresh_export_levels()

    def _on_slice_toggled(self, checked: bool) -> None:
        """Show the slice-geometry picker only while sliced inference is on."""
        if not hasattr(self, "row_slice_geometry"):
            return
        visible = self.combo_yolo_obb_mode.currentIndex() == 0 and bool(checked)
        self._set_widget_visible(self.row_slice_geometry, visible)
        self._set_widget_visible(self.row_slice_params, visible)
        if visible:
            self._on_slice_geometry_changed(self.combo_slice_geometry.currentIndex())

    def _on_slice_geometry_changed(self, _index: object) -> None:
        """Reveal only the SAHI parameters that apply to the chosen geometry mode.

        custom needs an explicit tile size (W/H); auto_object needs a target
        object fraction; auto_model derives the tile from the checkpoint and
        needs nothing. Tile overlap applies to every mode.
        """
        if not hasattr(self, "combo_slice_geometry"):
            return
        mode = self.combo_slice_geometry.currentText()
        is_custom = mode == "custom"
        is_auto_object = mode == "auto_object"
        self._set_widget_visible(getattr(self, "lbl_slice_tile_w", None), is_custom)
        self._set_widget_visible(getattr(self, "spin_slice_tile_w", None), is_custom)
        self._set_widget_visible(getattr(self, "lbl_slice_tile_h", None), is_custom)
        self._set_widget_visible(getattr(self, "spin_slice_tile_h", None), is_custom)
        self._set_widget_visible(
            getattr(self, "lbl_slice_object_fraction", None), is_auto_object
        )
        self._set_widget_visible(
            getattr(self, "spin_slice_object_fraction", None), is_auto_object
        )

    def _on_yolo_direct_task_changed(self, _index: object) -> object:
        """Sync the inferred-task label and fixed-angle row to the current task."""
        task = ["obb", "detect", "segment"][self.combo_yolo_direct_task.currentIndex()]
        if hasattr(self, "lbl_direct_task_inferred"):
            self.lbl_direct_task_inferred.setText(_DIRECT_TASK_LABELS[task])
        is_detect = task == "detect"
        self._set_widget_visible(getattr(self, "lbl_yolo_fixed_angle", None), is_detect)
        self._set_widget_visible(
            getattr(self, "spin_yolo_fixed_angle", None), is_detect
        )
        if hasattr(self._main_window, "_dataset_panel"):
            self._main_window._dataset_panel.refresh_export_levels()

    # =========================================================================
    # DIRECT MODEL TASK INFERENCE (from checkpoint / registry)
    # =========================================================================

    def _kick_direct_task_inference(self) -> None:
        """Schedule the background checkpoint-task inference (deferred).

        The actual work is deferred with ``QTimer.singleShot(0, ...)`` so it
        only starts once the application event loop is running. During MainWindow
        construction (and in tests, which build windows without an event loop)
        the timer never fires, so no checkpoint threads are spawned; in the app
        the inference starts on the next loop iteration, keeping the GUI free.
        """
        if self._task_kick_scheduled:
            return
        self._task_kick_scheduled = True
        QTimer.singleShot(0, self._run_scheduled_task_inference)

    def _run_scheduled_task_inference(self) -> None:
        """Infer the selected direct model's task and reflect it in the UI.

        Prefers the value already recorded in the registry (instant). Otherwise
        reads the checkpoint in a background worker so the GUI never freezes,
        then records the result in the registry for future sessions.
        """
        self._task_kick_scheduled = False
        if self._main_window is None:
            return
        try:
            model_path = str(self._main_window._get_selected_yolo_model_path() or "")
        except Exception:  # noqa: BLE001 - window may be mid-teardown
            return
        if not model_path:
            return
        from hydra_suite.core.inference.model_paths import (
            get_yolo_model_registered_task,
        )

        task = get_yolo_model_registered_task(model_path)
        if task in _DIRECT_TASK_INDEX:
            self._set_direct_task(task)
            return
        self._start_props_worker(model_path, "direct")

    def _kick_seq_crop_model_props(self) -> None:
        """Schedule the background read of the sequential crop-OBB model's
        trained input size (deferred to the event loop, like the task read).

        The registry ``training_params.imgsz`` (recorded at DetectKit training
        time) is the authoritative source; the checkpoint read only fills the
        gap for models imported without that metadata.
        """
        if self._seq_crop_kick_scheduled:
            return
        self._seq_crop_kick_scheduled = True
        QTimer.singleShot(0, self._run_scheduled_seq_crop_inference)

    def _run_scheduled_seq_crop_inference(self) -> None:
        self._seq_crop_kick_scheduled = False
        if self._main_window is None:
            return
        try:
            model_path = str(
                self._main_window._get_selected_yolo_crop_obb_model_path() or ""
            )
        except Exception:  # noqa: BLE001 - window may be mid-teardown
            return
        if not model_path:
            return
        if (
            self._seq_crop_worker is not None
            and getattr(self._seq_crop_worker, "_model_path", None) == model_path
        ):
            return  # already reading this exact model
        self._start_props_worker(model_path, "seq_crop_obb")

    def _start_props_worker(self, model_path: str, role: str) -> None:
        """Start a background checkpoint read for ``role`` (direct | seq_crop_obb).

        Bound-method receivers, not lambdas: a lambda connected before the main
        event loop starts (init / config-load populate) never receives its
        queued emission from the worker thread in PySide6, while QObject
        receivers do. The model path + role travel on the worker itself.
        """
        worker = _DirectTaskInferenceWorker(model_path)
        worker._role = role
        worker.props_inferred.connect(self._on_task_inferred)
        worker.finished.connect(self._on_task_worker_finished)
        if role == "seq_crop_obb":
            self._seq_crop_worker = worker
        else:
            self._task_worker = worker
        worker.start()

    def _on_task_inferred(self, task: str, imgsz: int) -> None:
        """Apply a background checkpoint-properties result if still current."""
        worker = self.sender()
        model_path = str(getattr(worker, "_model_path", "") or "")
        role = str(getattr(worker, "_role", "") or "")
        if role == "seq_crop_obb":
            self._apply_seq_crop_imgsz(model_path, imgsz)
            return
        if str(self._main_window._get_selected_yolo_model_path() or "") != model_path:
            return  # stale: the selection changed while the worker ran
        if task not in _DIRECT_TASK_INDEX:
            return  # unreadable checkpoint — keep the current (registry) value
        self._set_direct_task(task)
        try:
            from hydra_suite.core.inference.model_paths import register_yolo_model_task

            register_yolo_model_task(model_path, task)
        except Exception:  # noqa: BLE001 - backfill is best-effort
            logger.exception("Failed to backfill task metadata for %s", model_path)

    def _apply_seq_crop_imgsz(self, model_path: str, imgsz: int) -> None:
        """Auto-set stage-2 imgsz from the crop-OBB checkpoint when the
        registry has no training-recorded ``imgsz`` (manually imported models).

        The DetectKit-published ``training_params.imgsz`` wins whenever present;
        the checkpoint read is only a fallback so the user never has to type the
        model's own trained input size.
        """
        if imgsz <= 0:
            return
        try:
            from hydra_suite.core.inference.model_paths import get_yolo_model_metadata

            meta = get_yolo_model_metadata(model_path) or {}
            tp = meta.get("training_params")
            if isinstance(tp, dict) and tp.get("imgsz"):
                return  # training-recorded value is authoritative
        except Exception:  # noqa: BLE001 - best-effort
            return
        self.spin_yolo_seq_stage2_imgsz.setValue(int(imgsz))
        logger.info("Auto-set stage-2 imgsz=%d from checkpoint %s", imgsz, model_path)
        # Cache in the registry so subsequent selections are instant.
        try:
            from hydra_suite.core.inference.model_paths import (
                get_yolo_model_metadata,
                register_yolo_model,
            )

            meta = get_yolo_model_metadata(model_path) or {}
            tp = dict(meta.get("training_params") or {})
            tp["imgsz"] = int(imgsz)
            meta["training_params"] = tp
            register_yolo_model(model_path, meta)
        except Exception:  # noqa: BLE001 - backfill is best-effort
            logger.exception("Failed to backfill imgsz metadata for %s", model_path)
        self._sync_seq_advanced_derived_state()

    #: Sequential advanced knobs auto-derived from the models (field -> widget).
    _SEQ_AUTO_FIELD_WIDGETS = {
        "crop_pad": "spin_yolo_seq_crop_pad",
        "min_crop": "spin_yolo_seq_min_crop_px",
        "square": "chk_yolo_seq_square_crop",
        "imgsz": "spin_yolo_seq_stage2_imgsz",
    }

    def _sync_seq_advanced_derived_state(self) -> None:
        """Disable the sequential knobs whose values are auto-derived from the
        selected models, so auto-set values can't be silently diverged from.

        ``crop_pad`` / ``min_crop`` / ``square`` come from either sequential
        model's ``training_params`` (shared crop policy); ``imgsz`` from the
        crop model's ``training_params`` or the checkpoint fallback. Runtime
        knobs (stage-1 conf, crop batch, pow2 pad) stay editable — no model
        metadata records them. With no sequential models selected everything
        stays editable (manual fallback).
        """
        try:
            crop_path = str(
                self._main_window._get_selected_yolo_crop_obb_model_path() or ""
            )
            detect_path = str(
                self._main_window._get_selected_yolo_detect_model_path() or ""
            )
        except Exception:  # noqa: BLE001 - window may be mid-teardown
            return
        derived: set[str] = set()
        for path, is_crop in ((detect_path, False), (crop_path, True)):
            if not path:
                continue
            try:
                from hydra_suite.core.inference.model_paths import (
                    get_yolo_model_metadata,
                )

                tp = (get_yolo_model_metadata(path) or {}).get("training_params")
            except Exception:  # noqa: BLE001 - best-effort
                tp = None
            if not isinstance(tp, dict):
                continue
            if "crop_pad_ratio" in tp:
                derived.add("crop_pad")
            if "min_crop_size_px" in tp:
                derived.add("min_crop")
            if "enforce_square" in tp:
                derived.add("square")
            if is_crop and tp.get("imgsz"):
                derived.add("imgsz")
        for field, attr in self._SEQ_AUTO_FIELD_WIDGETS.items():
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.setEnabled(field not in derived)

    def _on_task_worker_finished(self) -> None:
        """Drop the finished worker reference so the next selection can start a new one.

        Only clears when the finishing worker is still the tracked one — a stale
        worker finishing after a newer one started must not clobber the newer
        worker's dedupe reference.
        """
        sender = self.sender()
        if sender is self._task_worker:
            self._task_worker = None
        elif sender is self._seq_crop_worker:
            self._seq_crop_worker = None

    def _set_direct_task(self, task: str) -> None:
        """Record an inferred task as the direct-model state (hidden combo)."""
        idx = _DIRECT_TASK_INDEX.get(task)
        if idx is None:
            return
        if self.combo_yolo_direct_task.currentIndex() != idx:
            self.combo_yolo_direct_task.setCurrentIndex(idx)  # refreshes label/UI
        else:
            self._on_yolo_direct_task_changed(idx)

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
                dialog_title="Add Direct Model",
            )
            return
        self._on_yolo_mode_changed(index)

    def apply_slice_meta_for_model(self, model_path: str) -> None:
        """Pre-fill SAHI settings from a model's .slice_meta.json sidecar, if present.

        Scale-independent trained knobs + the model-internal slice_trained_body_px;
        REFERENCE_BODY_SIZE (spin_reference_body_size) is deliberately left untouched
        (it is the full-frame tracking body size, a different quantity from the
        training-image body scale). No-op when no sidecar exists.
        """
        from hydra_suite.core.inference.slice_meta import (
            read_slice_meta,
            slice_meta_to_panel_values,
        )

        meta = read_slice_meta(model_path)
        if meta is None:
            return
        values = slice_meta_to_panel_values(meta)
        self.chk_slice_enabled.setChecked(bool(values["enabled"]))
        idx = self.combo_slice_geometry.findText(values["geometry_mode"])
        if idx >= 0:
            self.combo_slice_geometry.setCurrentIndex(idx)
        adv = self._main_window.advanced_config
        adv["slice_overlap"] = float(values["overlap"])
        adv["slice_object_tile_fraction"] = float(values["object_tile_fraction"])
        adv["slice_trained_body_px"] = float(values["trained_body_px"])
        # Display-sync the spins without letting their rounded values clobber
        # the exact metadata (e.g. 300/640 -> spin shows 0.47, config keeps
        # 0.46875).
        for spin, value in (
            (getattr(self, "spin_slice_overlap", None), values["overlap"]),
            (
                getattr(self, "spin_slice_object_fraction", None),
                values["object_tile_fraction"],
            ),
        ):
            if spin is None:
                continue
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)
        self._notify_matched_geometry()

    def _notify_matched_geometry(self) -> None:
        """Show a dismissible "Matched trained SAHI geometry" banner.

        Only for a user-driven model selection: during a config/preset restore
        the match is silent (logged), since the user did not just act.
        """
        main_window = self._main_window
        logger.info("Matched trained SAHI geometry from model sidecar")
        if getattr(main_window, "_restoring_config", False):
            return
        if hasattr(main_window, "statusBar"):
            try:
                main_window.statusBar().showMessage(
                    "Matched trained SAHI geometry", 5000
                )
            except Exception:
                pass

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
