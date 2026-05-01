"""ClassKitTrainingDialog — training dialog for ClassKit."""

from pathlib import Path
from statistics import median
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.classkit.config.custom_backbones import (
    custom_backbone_display_name,
    get_custom_backbone_choices,
    register_user_timm_backbones,
)
from hydra_suite.classkit.core.export.splits import (
    build_label_expansion_records,
    build_training_dataset_splits,
)
from hydra_suite.training.tiny_model import (
    describe_tiny_size_preset,
    get_tiny_size_preset_choices,
)
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811

from .timm_backbone_browser import TimmBackboneBrowserDialog


class ClassKitTrainingDialog(QDialog):
    """Training dialog for ClassKit: flat or multi-head, tiny CNN or YOLO-classify."""

    _AUTO_SIZE_SCALE_MIN = 0.25
    _AUTO_SIZE_SCALE_MAX = 4.0

    def __init__(
        self,
        scheme=None,
        n_labeled: int = 0,
        class_choices: Optional[List[str]] = None,
        labeled_label_names: Optional[List[str]] = None,
        initial_settings: Optional[dict] = None,
        recent_model_paths: Optional[List[str]] = None,
        average_image_size: Optional[Tuple[float, float]] = None,
        image_paths: Optional[List[Path]] = None,
        group_keys: Optional[List[str]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._scheme = scheme
        self._n_labeled = n_labeled
        self._class_choices = self._resolve_class_choices(class_choices)
        self._labeled_label_names = [
            str(label).strip()
            for label in (labeled_label_names or [])
            if str(label).strip()
        ]
        self._initial_settings = (
            dict(initial_settings) if isinstance(initial_settings, dict) else {}
        )
        self._recent_model_paths = self._normalize_recent_model_paths(
            recent_model_paths
        )
        self._average_image_size = average_image_size
        self._image_paths = [Path(path) for path in (image_paths or [])]
        self._group_keys = list(group_keys or [])
        self._used_split_group_fallback = False
        self._train_results = None
        self._worker = None
        self.setWindowTitle("Train Classifier")
        self.setMinimumWidth(640)
        self.setMinimumHeight(560)
        self._build_ui()
        self._apply_average_image_size_defaults()
        self._apply_initial_settings()

    @staticmethod
    def _nearest_computer_friendly_size(value: object) -> int:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 224
        rounded = int(((numeric / 32.0) + 0.5) // 1 * 32)
        return max(32, min(512, rounded))

    @staticmethod
    def _max_computer_friendly_size_without_upscale(value: object) -> int:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 224
        if numeric <= 32:
            return 32
        rounded = int(numeric // 32.0) * 32
        return max(32, min(512, rounded))

    @classmethod
    def _suggest_auto_input_sizes_for_dimensions(
        cls, dimensions: List[Tuple[float, float]]
    ) -> Optional[dict]:
        cleaned: List[Tuple[float, float]] = []
        for width, height in dimensions:
            try:
                width_f = float(width)
                height_f = float(height)
            except (TypeError, ValueError):
                continue
            if width_f > 0 and height_f > 0:
                cleaned.append((width_f, height_f))

        if not cleaned:
            return None

        widths = [width for width, _ in cleaned]
        heights = [height for _, height in cleaned]
        aspect_ratio = median(width / height for width, height in cleaned)
        max_width = max(widths)
        max_height = max(heights)

        if aspect_ratio >= 1.0:
            raw_width = max_width
            raw_height = min(max_height, raw_width / aspect_ratio)
        else:
            raw_height = max_height
            raw_width = min(max_width, raw_height * aspect_ratio)

        if raw_width <= 0:
            raw_width = max_width
        if raw_height <= 0:
            raw_height = max_height

        tiny_width = cls._max_computer_friendly_size_without_upscale(raw_width)
        tiny_height = cls._max_computer_friendly_size_without_upscale(raw_height)
        # TIMM backbones currently use a square resize, so preserve the
        # recommended long side to avoid over-downscaling elongated crops.
        square_size = cls._max_computer_friendly_size_without_upscale(
            max(tiny_width, tiny_height)
        )

        return {
            "tiny_width": tiny_width,
            "tiny_height": tiny_height,
            "custom_input_size": square_size,
            "aspect_ratio": float(aspect_ratio),
            "sample_count": len(cleaned),
        }

    @staticmethod
    def _set_combo_value(combo: QComboBox, value: object) -> None:
        if value is None:
            return
        idx = combo.findData(value)
        if idx < 0:
            idx = combo.findText(str(value))
        if idx >= 0:
            combo.setCurrentIndex(idx)

    @staticmethod
    def _set_spin_value(widget, settings: dict, key: str) -> None:
        if key not in settings:
            return
        try:
            widget.setValue(settings[key])
        except Exception:
            pass

    @staticmethod
    def _normalize_path_text(value: object) -> str:
        return str(value or "").strip()

    def _ensure_device_choice(self, value: object) -> None:
        device = str(value or "").strip().lower()
        if not device or self.device_combo.findData(device) >= 0:
            return

        labels = {
            "cpu": "CPU",
            "cuda": "CUDA GPU",
            "mps": "Apple Silicon (MPS)",
        }
        if device in labels:
            self.device_combo.addItem(labels[device], device)

    def _normalize_recent_model_paths(
        self, values: Optional[List[str]] = None
    ) -> List[str]:
        ordered: List[str] = []
        seen = set()
        for value in values or []:
            text = self._normalize_path_text(value)
            if not text:
                continue
            try:
                normalized = str(Path(text).expanduser().resolve())
            except Exception:
                normalized = text
            if normalized not in seen:
                seen.add(normalized)
                ordered.append(normalized)
        return ordered

    def _apply_average_image_size_defaults(self) -> None:
        if not self._average_image_size:
            return

        avg_width, avg_height = self._average_image_size
        width = self._nearest_computer_friendly_size(avg_width)
        height = self._nearest_computer_friendly_size(avg_height)
        square = self._nearest_computer_friendly_size((avg_width + avg_height) / 2.0)

        self.tiny_width_spin.setValue(width)
        self.tiny_height_spin.setValue(height)
        self._tiny_in_custom_width_spin.setValue(width)
        self._tiny_in_custom_height_spin.setValue(height)
        self._custom_input_size_spin.setValue(square)

    def _sample_image_dimensions(
        self, max_samples: int = 1024
    ) -> List[Tuple[float, float]]:
        sample_paths = list(self._image_paths)
        if not sample_paths:
            return []
        if len(sample_paths) > max_samples:
            step = (len(sample_paths) - 1) / float(max_samples - 1)
            sample_paths = [
                sample_paths[int(round(idx * step))] for idx in range(max_samples)
            ]

        dimensions: List[Tuple[float, float]] = []
        try:
            from PIL import Image
        except Exception:
            return []

        for path in sample_paths:
            try:
                with Image.open(path) as image:
                    width, height = image.size
            except Exception:
                continue
            if width > 0 and height > 0:
                dimensions.append((float(width), float(height)))
        return dimensions

    def _apply_auto_input_sizes(self, sizes: dict) -> None:
        tiny_width = int(sizes["tiny_width"])
        tiny_height = int(sizes["tiny_height"])
        custom_input_size = int(sizes["custom_input_size"])

        self.tiny_width_spin.setValue(tiny_width)
        self.tiny_height_spin.setValue(tiny_height)
        self._tiny_in_custom_width_spin.setValue(tiny_width)
        self._tiny_in_custom_height_spin.setValue(tiny_height)
        self._custom_input_size_spin.setValue(custom_input_size)

    @classmethod
    def _scale_auto_input_sizes(cls, sizes: dict, factor: object) -> dict:
        try:
            scale = float(factor)
        except (TypeError, ValueError):
            scale = 1.0
        scale = max(cls._AUTO_SIZE_SCALE_MIN, min(cls._AUTO_SIZE_SCALE_MAX, scale))

        scaled = dict(sizes)
        scaled["tiny_width"] = cls._nearest_computer_friendly_size(
            float(sizes["tiny_width"]) * scale
        )
        scaled["tiny_height"] = cls._nearest_computer_friendly_size(
            float(sizes["tiny_height"]) * scale
        )
        scaled["custom_input_size"] = cls._nearest_computer_friendly_size(
            float(sizes["custom_input_size"]) * scale
        )
        scaled["applied_rescale_factor"] = scale
        return scaled

    def _auto_set_sizes_from_images(self) -> None:
        dimensions = self._sample_image_dimensions()
        if not dimensions:
            QMessageBox.warning(
                self,
                "Auto Size Unavailable",
                "ClassKit could not read any valid image dimensions from this project.",
            )
            return

        sizes = self._suggest_auto_input_sizes_for_dimensions(dimensions)
        if not sizes:
            QMessageBox.warning(
                self,
                "Auto Size Unavailable",
                "ClassKit could not derive input sizes from the sampled images.",
            )
            return

        factor = self._auto_size_scale_spin.value()
        scaled_sizes = self._scale_auto_input_sizes(sizes, factor)
        self._apply_auto_input_sizes(scaled_sizes)
        self.append_log(
            "Auto-sized inputs from "
            f"{sizes['sample_count']:,} image(s): "
            f"Tiny {scaled_sizes['tiny_width']}x{scaled_sizes['tiny_height']}, "
            f"TIMM {scaled_sizes['custom_input_size']} square, "
            f"aspect ratio {sizes['aspect_ratio']:.2f}, "
            f"rescale x{scaled_sizes['applied_rescale_factor']:.2f}."
        )

    def _apply_initial_settings(self) -> None:
        settings = self._initial_settings
        if not settings:
            self._on_mode_changed()
            self._on_rebalance_mode_changed()
            self._on_custom_backbone_changed()
            self._refresh_tiny_preset_summary()
            self._refresh_data_summary()
            return

        self._ensure_device_choice(settings.get("device"))
        self._set_combo_value(
            self.mode_combo, self._normalize_mode_key(settings.get("mode"))
        )
        self._set_combo_value(self.device_combo, settings.get("device"))
        self._set_combo_value(self.base_model_combo, settings.get("base_model"))
        self._initial_model_path_edit.setText(
            self._normalize_path_text(settings.get("initial_model_path"))
        )
        self._set_combo_value(
            self.tiny_rebalance_combo, settings.get("tiny_rebalance_mode")
        )
        self._set_combo_value(
            self._custom_backbone_combo, settings.get("custom_backbone")
        )
        self._set_combo_value(
            self._custom_fine_tune_method_combo,
            settings.get("custom_fine_tune_method"),
        )
        self._set_spin_value(
            self._auto_size_scale_spin, settings, "auto_size_scale_factor"
        )
        self._set_combo_value(
            self.tiny_size_combo,
            settings.get("tiny_preset", "medium"),
        )
        self._set_combo_value(
            self._tiny_in_custom_size_combo,
            settings.get("tiny_preset", "medium"),
        )
        self._set_combo_value(
            self.split_strategy_combo, settings.get("split_strategy", "stratified")
        )

        for widget, key in (
            (self.epochs_spin, "epochs"),
            (self.batch_spin, "batch"),
            (self.lr_spin, "lr"),
            (
                self.prediction_confidence_threshold_spin,
                "prediction_confidence_threshold",
            ),
            (self.val_fraction_spin, "val_fraction"),
            (self.test_fraction_spin, "test_fraction"),
            (self.patience_spin, "patience"),
            (self.tiny_layers_spin, "tiny_layers"),
            (self.tiny_dim_spin, "tiny_dim"),
            (self.tiny_dropout_spin, "tiny_dropout"),
            (self.tiny_width_spin, "tiny_width"),
            (self.tiny_height_spin, "tiny_height"),
            (self.tiny_rebalance_power_spin, "tiny_rebalance_power"),
            (self.tiny_label_smoothing_spin, "tiny_label_smoothing"),
            (self._tiny_in_custom_width_spin, "tiny_width"),
            (self._tiny_in_custom_height_spin, "tiny_height"),
            (self._tiny_in_custom_layers_spin, "tiny_layers"),
            (self._tiny_in_custom_dim_spin, "tiny_dim"),
            (self._tiny_in_custom_dropout_spin, "tiny_dropout"),
            (self._custom_trainable_layers_spin, "custom_trainable_layers"),
            (self._custom_backbone_lr_spin, "custom_backbone_lr_scale"),
            (self._custom_layerwise_decay_spin, "custom_layerwise_lr_decay"),
            (
                self._custom_gradual_unfreeze_interval_spin,
                "custom_gradual_unfreeze_interval",
            ),
            (self._custom_input_size_spin, "custom_input_size"),
            (self.hue_spin, "hue"),
            (self.saturation_spin, "saturation"),
            (self.brightness_spin, "brightness"),
            (self.contrast_spin, "contrast"),
            (self.flip_ud_spin, "flipud"),
            (self.flip_lr_spin, "fliplr"),
        ):
            self._set_spin_value(widget, settings, key)
        if "monochrome" in settings:
            self.monochrome_check.setChecked(bool(settings.get("monochrome")))

        label_expansion = settings.get("label_expansion")
        if isinstance(label_expansion, dict) and label_expansion:
            self._exp_group.setChecked(True)
            for src_name, dst_name in label_expansion.get("fliplr", {}).items():
                self._add_lr_row(src_name, dst_name)
            for src_name, dst_name in label_expansion.get("flipud", {}).items():
                self._add_ud_row(src_name, dst_name)

        self._on_mode_changed()
        self._on_rebalance_mode_changed()
        self._on_custom_backbone_changed()
        self._refresh_tiny_preset_summary()
        self._refresh_data_summary()

    @staticmethod
    def _populate_tiny_size_combo(combo: QComboBox) -> None:
        combo.clear()
        for key, label in get_tiny_size_preset_choices():
            combo.addItem(label, key)

    def _refresh_tiny_preset_summary(self) -> None:
        self._tiny_size_summary_label.setText(
            describe_tiny_size_preset(self.tiny_size_combo.currentData())
        )
        self._tiny_in_custom_size_summary_label.setText(
            describe_tiny_size_preset(self._tiny_in_custom_size_combo.currentData())
        )

    def _sync_tiny_preset_controls(self, source: QComboBox) -> None:
        preset = str(source.currentData() or "medium")
        for combo in (self.tiny_size_combo, self._tiny_in_custom_size_combo):
            if combo is source:
                continue
            if str(combo.currentData() or "") == preset:
                continue
            combo.blockSignals(True)
            self._set_combo_value(combo, preset)
            combo.blockSignals(False)
        self._refresh_tiny_preset_summary()

    def _resolve_class_choices(
        self, class_choices: Optional[List[str]] = None
    ) -> List[str]:
        ordered: List[str] = []

        def _add(value: object) -> None:
            text = str(value).strip()
            if text and text not in ordered:
                ordered.append(text)

        for value in class_choices or []:
            _add(value)

        if self._scheme is not None:
            factors = getattr(self._scheme, "factors", None) or []
            if len(factors) == 1:
                for label in getattr(factors[0], "labels", []) or []:
                    _add(label)

        return ordered

    @staticmethod
    def _normalize_mode_key(value: object) -> str:
        mode = str(value or "").strip()
        if mode == "flat_tiny":
            return "flat_custom"
        if mode == "multihead_tiny":
            return "multihead_custom"
        return mode

    def _fallback_mode_keys(self) -> list[str]:
        factor_count = len(getattr(self._scheme, "factors", []) or [])
        if factor_count > 1:
            return [
                "flat_custom",
                "flat_yolo",
                "multihead_yolo",
                "multihead_custom",
                "multihead_custom_shared",
            ]
        return ["flat_custom", "flat_yolo"]

    def _resolved_mode_keys(self) -> list[str]:
        if self._scheme is None:
            return ["flat_custom", "flat_yolo"]

        normalized_modes = [
            self._normalize_mode_key(mode)
            for mode in getattr(self._scheme, "training_modes", []) or []
        ]
        if not normalized_modes:
            normalized_modes = self._fallback_mode_keys()

        ordered: list[str] = []
        seen = set()
        for key in [
            "flat_custom",
            "flat_yolo",
            "multihead_yolo",
            "multihead_custom",
            "multihead_custom_shared",
        ]:
            if key in normalized_modes and key not in seen:
                ordered.append(key)
                seen.add(key)
        return ordered

    @staticmethod
    def _mode_parts(mode: object) -> tuple[str, str]:
        normalized = str(mode or "").strip()
        structure = "multihead" if normalized.startswith("multihead") else "flat"
        family = "yolo" if "yolo" in normalized else "custom"
        return structure, family

    def _resolved_mode_structures(self) -> list[str]:
        structures: list[str] = []
        seen = set()
        for key in self._resolved_mode_keys():
            structure, _family = self._mode_parts(key)
            if structure not in seen:
                structures.append(structure)
                seen.add(structure)
        return structures

    def _resolved_mode_families(self, structure: object) -> list[str]:
        normalized_structure = str(structure or "").strip()
        families: list[str] = []
        seen = set()
        for family in ["custom", "yolo"]:
            mode_key = f"{normalized_structure}_{family}"
            if mode_key in self._resolved_mode_keys() and family not in seen:
                families.append(family)
                seen.add(family)
        return families

    def _populate_mode_family_combo(self, structure: object, preferred: object) -> None:
        current_family = str(preferred or "").strip()
        family_labels = {
            "custom": "Custom CNN",
            "yolo": "YOLO-classify",
        }
        available_families = self._resolved_mode_families(structure)
        selected_family = (
            current_family if current_family in available_families else ""
        ) or (available_families[0] if available_families else "")

        self.mode_family_combo.blockSignals(True)
        self.mode_family_combo.clear()
        for family in available_families:
            self.mode_family_combo.addItem(family_labels[family], family)
        self._set_combo_value(self.mode_family_combo, selected_family)
        self.mode_family_combo.blockSignals(False)

    def _sync_mode_controls_from_mode(self, mode: object = None) -> None:
        normalized_mode = self._normalize_mode_key(
            mode or self.mode_combo.currentData()
        )
        preferred_structure, preferred_family = self._mode_parts(normalized_mode)
        structure_labels = {
            "flat": "Flat labels",
            "multihead": "Multi-head per factor",
        }
        structures = self._resolved_mode_structures()
        selected_structure = (
            preferred_structure if preferred_structure in structures else ""
        ) or (structures[0] if structures else "")

        self.mode_structure_combo.blockSignals(True)
        self.mode_structure_combo.clear()
        for structure in structures:
            self.mode_structure_combo.addItem(structure_labels[structure], structure)
        self._set_combo_value(self.mode_structure_combo, selected_structure)
        self.mode_structure_combo.blockSignals(False)

        self._populate_mode_family_combo(selected_structure, preferred_family)

    def _sync_mode_combo_from_controls(self) -> None:
        structure = str(self.mode_structure_combo.currentData() or "").strip()
        family = str(self.mode_family_combo.currentData() or "").strip()
        if not structure or not family:
            return

        target_mode = f"{structure}_{family}"
        index = self.mode_combo.findData(target_mode)
        if index < 0:
            return
        if self.mode_combo.currentIndex() != index:
            self.mode_combo.setCurrentIndex(index)

    def _on_mode_structure_changed(self) -> None:
        structure = self.mode_structure_combo.currentData()
        preferred_family = self.mode_family_combo.currentData()
        self._populate_mode_family_combo(structure, preferred_family)
        self._sync_mode_combo_from_controls()

    def _on_internal_mode_changed(self) -> None:
        self._sync_mode_controls_from_mode()
        self._on_mode_changed()

    def _build_general_tab(self):
        """Build the General settings tab and return its widget."""
        self.general_tab = QWidget()
        form = QFormLayout()
        form.setSpacing(8)

        self.mode_combo = QComboBox()
        self._populate_modes()
        self.mode_combo.setVisible(False)

        self.mode_structure_combo = QComboBox()
        self.mode_structure_combo.setToolTip(
            "Choose whether to train one classifier over flat labels or one classifier per scheme factor."
        )
        form.addRow("<b>Label Layout:</b>", self.mode_structure_combo)

        self.mode_family_combo = QComboBox()
        self.mode_family_combo.setToolTip(
            "Choose the training backend for the selected label layout."
        )
        form.addRow("<b>Model Family:</b>", self.mode_family_combo)
        self._sync_mode_controls_from_mode()

        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU", "cpu")
        try:
            import torch

            if torch.cuda.is_available():
                self.device_combo.addItem("CUDA GPU", "cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device_combo.addItem("Apple Silicon (MPS)", "mps")
                self.device_combo.setCurrentIndex(self.device_combo.count() - 1)
        except Exception:
            pass
        self.device_combo.setToolTip(
            "Hardware device used for model training (PyTorch backend)."
        )
        form.addRow("<b>Training Device:</b>", self.device_combo)

        self._build_base_model_combo(form)
        self._build_initial_model_selector(form)
        self._build_hyperparams_widgets(form)

        self._mode_desc_label = QLabel("")
        self._mode_desc_label.setWordWrap(True)
        self._mode_desc_label.setStyleSheet(
            "color: #ffffff; font-size: 11px; padding: 4px 8px; background: #1e1e1e; border-radius: 4px;"
        )
        form.addRow("", self._mode_desc_label)

        self._data_summary_group = QGroupBox("Dataset Summary")
        data_summary_layout = QVBoxLayout(self._data_summary_group)
        data_summary_layout.setContentsMargins(10, 8, 10, 8)
        self._data_summary_label = QLabel("")
        self._data_summary_label.setWordWrap(True)
        self._data_summary_label.setStyleSheet("color:#ffffff; font-size:11px;")
        data_summary_layout.addWidget(self._data_summary_label)

        self._sample_preview_group = QGroupBox("Training Sample Preview")
        sample_preview_layout = QVBoxLayout(self._sample_preview_group)
        sample_preview_layout.setContentsMargins(10, 8, 10, 8)
        self._sample_preview_note = QLabel(
            "Representative labeled source images before augmentation. Captions show the current label, split, and filename."
        )
        self._sample_preview_note.setWordWrap(True)
        self._sample_preview_note.setStyleSheet("color:#cfcfcf; font-size:11px;")
        sample_preview_layout.addWidget(self._sample_preview_note)

        preview_toggle_row = QHBoxLayout()
        preview_toggle_row.setContentsMargins(0, 0, 0, 0)
        self._sample_preview_monochrome_toggle = QCheckBox("Preview monochrome samples")
        self._sample_preview_monochrome_toggle.setToolTip(
            "When monochrome mode is enabled for training, render these sample thumbnails in grayscale so the preview matches the derived training dataset."
        )
        self._sample_preview_monochrome_toggle.setEnabled(False)
        preview_toggle_row.addWidget(self._sample_preview_monochrome_toggle)
        preview_toggle_row.addStretch()
        sample_preview_layout.addLayout(preview_toggle_row)

        self._sample_preview_scroll = QScrollArea()
        self._sample_preview_scroll.setWidgetResizable(True)
        self._sample_preview_scroll.setFrameShape(QFrame.NoFrame)
        self._sample_preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._sample_preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._sample_preview_scroll.setMinimumHeight(220)

        self._sample_preview_container = QWidget()
        self._sample_preview_cards_layout = QHBoxLayout(self._sample_preview_container)
        self._sample_preview_cards_layout.setContentsMargins(0, 0, 0, 0)
        self._sample_preview_cards_layout.setSpacing(8)
        self._sample_preview_scroll.setWidget(self._sample_preview_container)
        sample_preview_layout.addWidget(self._sample_preview_scroll)

        scroll_content = QWidget()
        layout_gen = QVBoxLayout(scroll_content)
        layout_gen.setContentsMargins(0, 0, 0, 0)
        layout_gen.addLayout(form)
        layout_gen.addWidget(self._data_summary_group)
        layout_gen.addWidget(self._sample_preview_group)
        layout_gen.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidget(scroll_content)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        outer_layout = QVBoxLayout(self.general_tab)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll_area)
        return self.general_tab

    def _build_auto_size_row(self, layout) -> None:
        self._auto_size_btn = QPushButton("Auto Set Size From Images")
        self._auto_size_btn.setToolTip(
            "Inspect the current project's image sizes and choose computer-friendly "
            "Tiny CNN width/height plus a square TIMM size based on the dataset's "
            "maximum extent and typical aspect ratio."
        )
        self._auto_size_btn.setEnabled(bool(self._image_paths))
        self._auto_size_btn.clicked.connect(self._auto_set_sizes_from_images)

        self._auto_size_helper_label = QLabel(
            "Uses sampled project images to minimize resizing loss without unnecessary upscaling."
        )
        self._auto_size_helper_label.setWordWrap(True)
        self._auto_size_helper_label.setStyleSheet("color: #ffffff; font-size: 11px;")

        self._auto_size_controls = QWidget()
        row_layout = QVBoxLayout(self._auto_size_controls)
        row_layout.setContentsMargins(0, 6, 0, 0)
        row_layout.setSpacing(4)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.addWidget(self._auto_size_btn)

        self._auto_size_scale_label = QLabel("Rescale:")
        self._auto_size_scale_spin = QDoubleSpinBox()
        self._auto_size_scale_spin.setDecimals(2)
        self._auto_size_scale_spin.setRange(
            self._AUTO_SIZE_SCALE_MIN, self._AUTO_SIZE_SCALE_MAX
        )
        self._auto_size_scale_spin.setSingleStep(0.25)
        self._auto_size_scale_spin.setValue(1.0)
        self._auto_size_scale_spin.setSuffix("x")
        self._auto_size_scale_spin.setToolTip(
            "Scale the auto-detected sizes before applying them to the Tiny and custom backbone inputs."
        )
        button_row.addWidget(self._auto_size_scale_label)
        button_row.addWidget(self._auto_size_scale_spin)
        button_row.addStretch()

        row_layout.addWidget(QLabel("<b>Input Size Helper</b>"))
        row_layout.addLayout(button_row)
        row_layout.addWidget(self._auto_size_helper_label)
        layout.addWidget(self._auto_size_controls)

    def _build_base_model_combo(self, form):
        """Build the YOLO base model combo box."""
        self.base_model_combo = QComboBox()
        for m in [
            "yolo26n-cls.pt",
            "yolo26s-cls.pt",
            "yolo26m-cls.pt",
            "yolo26l-cls.pt",
            "yolo26x-cls.pt",
        ]:
            self.base_model_combo.addItem(m, m)
        self.base_model_combo.setToolTip(
            "Pretrained YOLO backbone to fine-tune.\n"
            "'n' = nano (fastest, least RAM)  'x' = extra-large (most accurate, needs GPU)"
        )
        self._base_model_row_label = QLabel("<b>Base Model (YOLO):</b>")
        form.addRow(self._base_model_row_label, self.base_model_combo)

    def _build_initial_model_selector(self, form) -> None:
        self._initial_model_row_label = QLabel("<b>Start From Previous Model:</b>")
        self._initial_model_path_edit = QLineEdit()
        self._initial_model_path_edit.setReadOnly(True)

        self._initial_model_browse_btn = QPushButton("Browse...")
        self._initial_model_recent_btn = QPushButton("Recent")
        self._initial_model_clear_btn = QPushButton("Clear")

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        row_layout.addWidget(self._initial_model_path_edit, 1)
        row_layout.addWidget(self._initial_model_browse_btn)
        row_layout.addWidget(self._initial_model_recent_btn)
        row_layout.addWidget(self._initial_model_clear_btn)

        self._initial_model_browse_btn.clicked.connect(self._browse_initial_model)
        self._initial_model_recent_btn.clicked.connect(
            self._choose_recent_initial_model
        )
        self._initial_model_clear_btn.clicked.connect(
            lambda: self._initial_model_path_edit.clear()
        )

        self._initial_model_row = row
        form.addRow(self._initial_model_row_label, row)

    def _compatible_initial_model_suffixes(self) -> tuple[str, ...]:
        mode = self._normalize_mode_key(self.mode_combo.currentData())
        if "yolo" in mode:
            return (".pt",)
        return (".pth",)

    def _compatible_recent_model_paths(self) -> List[str]:
        suffixes = self._compatible_initial_model_suffixes()
        return [
            path for path in self._recent_model_paths if path.lower().endswith(suffixes)
        ]

    def _browse_initial_model(self) -> None:
        mode = self._normalize_mode_key(self.mode_combo.currentData())
        if "yolo" in mode:
            title = "Select YOLO Starting Weights"
            filter_text = "YOLO Weights (*.pt);;All Files (*)"
        else:
            title = "Select ClassKit Starting Checkpoint"
            filter_text = "ClassKit Checkpoints (*.pth);;All Files (*)"

        start_dir = self._normalize_path_text(self._initial_model_path_edit.text())
        path, _selected = QFileDialog.getOpenFileName(
            self,
            title,
            start_dir,
            filter_text,
        )
        if path:
            self._initial_model_path_edit.setText(path)

    def _choose_recent_initial_model(self) -> None:
        choices = self._compatible_recent_model_paths()
        if not choices:
            QMessageBox.information(
                self,
                "No Recent Models",
                "No compatible trainable model artifacts were found in this project's history.",
            )
            return
        selected, accepted = QInputDialog.getItem(
            self,
            "Recent Project Models",
            "Choose a previous exported model to use as the starting point:",
            choices,
            0,
            False,
        )
        if accepted and selected:
            self._initial_model_path_edit.setText(str(selected))

    def _build_hyperparams_widgets(self, form):
        """Build epoch, batch, lr, split, val fraction, and patience widgets."""
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 999999)
        self.epochs_spin.setValue(50)
        form.addRow("<b>Epochs:</b>", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(4, 512)
        self.batch_spin.setValue(32)
        form.addRow("<b>Batch Size:</b>", self.batch_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001)
        form.addRow("<b>Learning Rate:</b>", self.lr_spin)

        self.split_strategy_combo = QComboBox()
        self.split_strategy_combo.addItem("Stratified", "stratified")
        self.split_strategy_combo.addItem("Random", "random")
        form.addRow("<b>Split Strategy:</b>", self.split_strategy_combo)

        self.val_fraction_spin = QDoubleSpinBox()
        self.val_fraction_spin.setRange(0.0, 0.5)
        self.val_fraction_spin.setSingleStep(0.05)
        self.val_fraction_spin.setDecimals(2)
        self.val_fraction_spin.setValue(0.2)
        form.addRow("<b>Val Fraction:</b>", self.val_fraction_spin)

        self.test_fraction_spin = QDoubleSpinBox()
        self.test_fraction_spin.setRange(0.0, 0.5)
        self.test_fraction_spin.setSingleStep(0.05)
        self.test_fraction_spin.setDecimals(2)
        self.test_fraction_spin.setValue(0.0)
        form.addRow("<b>Test Fraction:</b>", self.test_fraction_spin)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 500)
        self.patience_spin.setValue(10)
        form.addRow("<b>Patience:</b>", self.patience_spin)

        self.epochs_spin.setToolTip(
            "Training epochs. More = slower but potentially better.\nDefault: 50."
        )
        self.batch_spin.setToolTip(
            "Mini-batch size.\nTiny CNN: 32 typical.  YOLO: 16 on GPU, 8 on CPU."
        )
        self.lr_spin.setToolTip(
            "Initial learning rate. 0.001 is a robust default for both Tiny and YOLO."
        )

        self.prediction_confidence_threshold_spin = QDoubleSpinBox()
        self.prediction_confidence_threshold_spin.setRange(0.0, 1.0)
        self.prediction_confidence_threshold_spin.setSingleStep(0.05)
        self.prediction_confidence_threshold_spin.setDecimals(2)
        self.prediction_confidence_threshold_spin.setValue(0.5)
        self.prediction_confidence_threshold_spin.setToolTip(
            "Inference-time abstention threshold for predictions from this model. "
            "Predictions below this confidence are treated as unknown in ClassKit and recommended as unknown for TrackerKit consumers."
        )
        form.addRow(
            "<b>Prediction Confidence Threshold:</b>",
            self.prediction_confidence_threshold_spin,
        )

        self.split_strategy_combo.setToolTip(
            "How to pick train and validation items. Stratified preserves class balance; Random uses a deterministic shuffled split."
        )
        self.val_fraction_spin.setToolTip(
            "Fraction of labeled images reserved for validation (0.2 = 20%).\n"
            "Set to 0 only if you have very few labels."
        )
        self.test_fraction_spin.setToolTip(
            "Optional fraction of labeled images reserved for a final test split.\n"
            "Training still uses train/val; when test > 0, post-training metrics prefer the held-out test split."
        )
        self.patience_spin.setToolTip(
            "Early stopping: halt if val accuracy doesn't improve for N consecutive epochs.\n"
            "Set to 0 to disable early stopping."
        )

    def _current_split_strategy(self) -> str:
        return str(self.split_strategy_combo.currentData() or "stratified")

    @staticmethod
    def _safe_class_text(value: object) -> str:
        if value is None or isinstance(value, bool):
            return ""
        return str(value).strip()

    def _build_class_combo(self, initial_text: object = "") -> QComboBox:
        combo = QComboBox()
        combo.setMinimumWidth(160)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)

        if self._class_choices:
            for cls in self._class_choices:
                combo.addItem(cls, cls)
        else:
            combo.addItem("(no classes)", "")
            combo.setEnabled(False)

        initial = self._safe_class_text(initial_text)
        if initial:
            idx = combo.findData(initial)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        return combo

    def _add_reverse_mapping(
        self,
        rows: List[Tuple[QComboBox, QComboBox]],
        add_row,
        src: QComboBox,
        dst: QComboBox,
    ) -> None:
        src_name = self._safe_class_text(src.currentData() or src.currentText())
        dst_name = self._safe_class_text(dst.currentData() or dst.currentText())
        if not src_name or not dst_name or src_name == dst_name:
            return
        for row_src, row_dst in rows:
            row_src_name = self._safe_class_text(
                row_src.currentData() or row_src.currentText()
            )
            row_dst_name = self._safe_class_text(
                row_dst.currentData() or row_dst.currentText()
            )
            if row_src_name == dst_name and row_dst_name == src_name:
                return
        add_row(dst_name, src_name)

    def _add_mapping_row(
        self,
        rows: List[Tuple[QComboBox, QComboBox]],
        rows_layout: QVBoxLayout,
        add_row,
        src_text: str = "",
        dst_text: str = "",
    ) -> None:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        src = self._build_class_combo(src_text)
        dst = self._build_class_combo(dst_text)
        add_rev = QPushButton("Reverse")
        add_rev.setFixedWidth(72)
        add_rev.setToolTip("Add the reverse mapping (destination \u2192 source)")
        remove_btn = QPushButton("Remove")
        remove_btn.setFixedWidth(72)
        remove_btn.setStyleSheet("color:#c66;")

        row_layout.addWidget(src, 1)
        row_layout.addWidget(QLabel("\u2192"))
        row_layout.addWidget(dst, 1)
        row_layout.addWidget(add_rev)
        row_layout.addWidget(remove_btn)
        rows_layout.addWidget(row_widget)

        pair = (src, dst)
        rows.append(pair)
        src.currentTextChanged.connect(lambda _text: self._refresh_data_summary())
        dst.currentTextChanged.connect(lambda _text: self._refresh_data_summary())

        def _remove_row() -> None:
            if pair in rows:
                rows.remove(pair)
            row_widget.setParent(None)
            self._refresh_data_summary()

        remove_btn.clicked.connect(_remove_row)
        add_rev.clicked.connect(
            lambda: self._add_reverse_mapping(rows, add_row, src, dst)
        )
        self._refresh_data_summary()

    def _build_mapping_section(
        self,
        layout: QVBoxLayout,
        title: str,
        add_row,
    ) -> tuple[QWidget, QVBoxLayout]:
        header_row = QHBoxLayout()
        header_row.addWidget(QLabel(title))
        add_btn = QPushButton("+ Add pair")
        add_btn.setFixedWidth(90)
        add_btn.clicked.connect(lambda _checked=False: add_row())
        header_row.addWidget(add_btn)
        header_row.addStretch()
        layout.addLayout(header_row)

        rows_widget = QWidget()
        rows_layout = QVBoxLayout(rows_widget)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        rows_layout.setSpacing(3)
        layout.addWidget(rows_widget)
        return rows_widget, rows_layout

    def _build_expansion_group(self) -> QGroupBox:
        exp_group = QGroupBox("Label-Switching Expansion  (advanced)")
        exp_group.setCheckable(True)
        exp_group.setChecked(False)
        exp_group.setToolTip(
            "Generate deterministic mirrored copies of training images with remapped labels.\n"
            "Useful when a flip transforms one class into another (e.g. 'left' \u2194 'right'\n"
            "on a horizontal flip).  Expanded copies are only added to the train split."
        )
        exp_vbox = QVBoxLayout(exp_group)

        exp_note = QLabel(
            "<i>Each row: source label \u2192 destination label when that flip is applied.<br>"
            "Pairs are applied to train images only; evaluation data is never flipped.<br>"
            "When enabled, random flip augmentation is disabled; hue, saturation, brightness, contrast, and monochrome settings remain available.</i>"
        )
        exp_note.setWordWrap(True)
        exp_note.setStyleSheet("color:#ffffff; font-size:11px;")
        exp_vbox.addWidget(exp_note)

        self._exp_constraints_label = QLabel("")
        self._exp_constraints_label.setWordWrap(True)
        self._exp_constraints_label.setStyleSheet("color:#e0c070; font-size:11px;")
        exp_vbox.addWidget(self._exp_constraints_label)

        self._lr_mapping_rows = []
        self._ud_mapping_rows = []

        def _add_lr_row(src_text="", dst_text=""):
            self._add_mapping_row(
                self._lr_mapping_rows,
                self._lr_rows_layout,
                _add_lr_row,
                src_text,
                dst_text,
            )

        self._lr_rows_widget, self._lr_rows_layout = self._build_mapping_section(
            exp_vbox,
            "<b>Horizontal flip (LR) mappings:</b>",
            _add_lr_row,
        )
        self._add_lr_row = _add_lr_row

        def _add_ud_row(src_text="", dst_text=""):
            self._add_mapping_row(
                self._ud_mapping_rows,
                self._ud_rows_layout,
                _add_ud_row,
                src_text,
                dst_text,
            )

        self._ud_rows_widget, self._ud_rows_layout = self._build_mapping_section(
            exp_vbox,
            "<b>Vertical flip (UD) mappings:</b>",
            _add_ud_row,
        )
        self._add_ud_row = _add_ud_row
        return exp_group

    def _build_tiny_architecture_tab(self) -> None:
        self.tiny_tab = QWidget()
        tiny_layout = QVBoxLayout(self.tiny_tab)
        tiny_form = QFormLayout()

        self.tiny_size_combo = QComboBox()
        self._populate_tiny_size_combo(self.tiny_size_combo)
        tiny_form.addRow("<b>Tiny Size:</b>", self.tiny_size_combo)

        self._tiny_size_summary_label = QLabel()
        self._tiny_size_summary_label.setWordWrap(True)
        self._tiny_size_summary_label.setStyleSheet("color: #666;")
        tiny_form.addRow("", self._tiny_size_summary_label)

        self.tiny_layers_spin = QSpinBox()
        self.tiny_layers_spin.setRange(1, 10)
        self.tiny_layers_spin.setValue(1)
        tiny_form.addRow("<b>Hidden Layers:</b>", self.tiny_layers_spin)

        self.tiny_dim_spin = QSpinBox()
        self.tiny_dim_spin.setRange(16, 1024)
        self.tiny_dim_spin.setValue(96)
        self.tiny_dim_spin.setSingleStep(16)
        tiny_form.addRow("<b>Hidden Dim:</b>", self.tiny_dim_spin)

        self.tiny_dropout_spin = QDoubleSpinBox()
        self.tiny_dropout_spin.setRange(0.0, 0.9)
        self.tiny_dropout_spin.setValue(0.1)
        self.tiny_dropout_spin.setSingleStep(0.05)
        tiny_form.addRow("<b>Dropout:</b>", self.tiny_dropout_spin)

        self.tiny_width_spin = QSpinBox()
        self.tiny_width_spin.setRange(32, 512)
        self.tiny_width_spin.setValue(128)
        self.tiny_width_spin.setSingleStep(32)
        tiny_form.addRow("<b>Input Width:</b>", self.tiny_width_spin)

        self.tiny_height_spin = QSpinBox()
        self.tiny_height_spin.setRange(32, 512)
        self.tiny_height_spin.setValue(64)
        self.tiny_height_spin.setSingleStep(32)
        tiny_form.addRow("<b>Input Height:</b>", self.tiny_height_spin)

        self.tiny_label_smoothing_spin = QDoubleSpinBox()
        self.tiny_label_smoothing_spin.setRange(0.0, 0.4)
        self.tiny_label_smoothing_spin.setSingleStep(0.01)
        self.tiny_label_smoothing_spin.setValue(0.0)
        self.tiny_label_smoothing_spin.setDecimals(2)
        tiny_form.addRow("<b>Label Smoothing:</b>", self.tiny_label_smoothing_spin)

        self.tiny_layers_spin.setToolTip("Number of hidden layers in the MLP head.")
        self.tiny_dim_spin.setToolTip(
            "Neurons per hidden layer. Larger = more capacity, slower."
        )
        self.tiny_dropout_spin.setToolTip(
            "Dropout regularization probability (0 = disabled)."
        )
        self.tiny_width_spin.setToolTip(
            "Input image width (px). Match to your typical crop size."
        )
        self.tiny_height_spin.setToolTip(
            "Input image height (px). Match to your typical crop size."
        )
        self.tiny_label_smoothing_spin.setToolTip(
            "Cross-entropy label smoothing for Tiny CNN training (0.0 = disabled)."
        )
        self.tiny_size_combo.setToolTip(
            "Backbone capacity preset for the tiny classifier.\n"
            "Tiny-S is fastest, Tiny-M is the default balance, Tiny-L is the strongest lightweight option."
        )

        tiny_layout.addLayout(tiny_form)
        tiny_layout.addStretch()
        self._tiny_tab_idx = self.tabs.addTab(self.tiny_tab, "Tiny Architecture")

    def _build_custom_cnn_tab(self) -> None:
        self.custom_tab = QScrollArea()
        self.custom_tab.setWidgetResizable(True)
        self.custom_tab.setFrameShape(QFrame.NoFrame)

        custom_content = QWidget()
        custom_form = QFormLayout(custom_content)
        custom_form.setContentsMargins(0, 0, 0, 0)
        custom_form.setSpacing(8)
        self.custom_tab.setWidget(custom_content)

        self._custom_backbone_combo = QComboBox()
        self._reload_custom_backbone_choices(selected="tinyclassifier")
        custom_form.addRow("Backbone:", self._custom_backbone_combo)

        self._custom_add_timm_btn = QPushButton("Add from TIMM...")
        self._custom_add_timm_btn.setToolTip(
            "Browse valid pretrained TIMM image-classification backbones and add them to this picker."
        )
        custom_form.addRow("", self._custom_add_timm_btn)

        custom_layout = custom_content.layout()
        self._build_auto_size_row(custom_layout)

        self._tiny_in_custom_width_label = QLabel("Input width")
        self._tiny_in_custom_width_spin = QSpinBox()
        self._tiny_in_custom_width_spin.setRange(32, 512)
        self._tiny_in_custom_width_spin.setValue(128)
        custom_form.addRow(
            self._tiny_in_custom_width_label, self._tiny_in_custom_width_spin
        )

        self._tiny_in_custom_size_label = QLabel("Tiny size")
        self._tiny_in_custom_size_combo = QComboBox()
        self._populate_tiny_size_combo(self._tiny_in_custom_size_combo)
        custom_form.addRow(
            self._tiny_in_custom_size_label, self._tiny_in_custom_size_combo
        )

        self._tiny_in_custom_size_summary_label = QLabel()
        self._tiny_in_custom_size_summary_label.setWordWrap(True)
        self._tiny_in_custom_size_summary_label.setStyleSheet("color: #666;")
        custom_form.addRow("", self._tiny_in_custom_size_summary_label)

        self._tiny_in_custom_height_label = QLabel("Input height")
        self._tiny_in_custom_height_spin = QSpinBox()
        self._tiny_in_custom_height_spin.setRange(32, 512)
        self._tiny_in_custom_height_spin.setValue(64)
        custom_form.addRow(
            self._tiny_in_custom_height_label, self._tiny_in_custom_height_spin
        )

        self._tiny_in_custom_layers_label = QLabel("Hidden layers")
        self._tiny_in_custom_layers_spin = QSpinBox()
        self._tiny_in_custom_layers_spin.setRange(0, 4)
        self._tiny_in_custom_layers_spin.setValue(1)
        custom_form.addRow(
            self._tiny_in_custom_layers_label, self._tiny_in_custom_layers_spin
        )

        self._tiny_in_custom_dim_label = QLabel("Hidden dim")
        self._tiny_in_custom_dim_spin = QSpinBox()
        self._tiny_in_custom_dim_spin.setRange(16, 512)
        self._tiny_in_custom_dim_spin.setValue(96)
        custom_form.addRow(
            self._tiny_in_custom_dim_label, self._tiny_in_custom_dim_spin
        )

        self._tiny_in_custom_dropout_label = QLabel("Dropout")
        self._tiny_in_custom_dropout_spin = QDoubleSpinBox()
        self._tiny_in_custom_dropout_spin.setRange(0.0, 0.9)
        self._tiny_in_custom_dropout_spin.setSingleStep(0.05)
        self._tiny_in_custom_dropout_spin.setValue(0.1)
        custom_form.addRow(
            self._tiny_in_custom_dropout_label, self._tiny_in_custom_dropout_spin
        )

        self._custom_fine_tune_method_label = QLabel("Fine-tuning")
        self._custom_fine_tune_method_combo = QComboBox()
        self._custom_fine_tune_method_combo.addItem("Head only", "head_only")
        self._custom_fine_tune_method_combo.addItem("Full fine-tune", "full_finetune")
        self._custom_fine_tune_method_combo.addItem(
            "Partial unfreezing", "partial_unfreezing"
        )
        self._custom_fine_tune_method_combo.addItem(
            "Layerwise LR decay", "layerwise_lr_decay"
        )
        self._custom_fine_tune_method_combo.addItem(
            "Gradual unfreezing", "gradual_unfreezing"
        )
        custom_form.addRow(
            self._custom_fine_tune_method_label,
            self._custom_fine_tune_method_combo,
        )

        self._custom_trainable_layers_label = QLabel("Unfreeze last N stages")
        self._custom_trainable_layers_spin = QSpinBox()
        self._custom_trainable_layers_spin.setRange(1, 16)
        self._custom_trainable_layers_spin.setValue(1)
        self._custom_trainable_layers_spin.setToolTip(
            "For partial unfreezing, train the head plus the last N backbone stages."
        )
        custom_form.addRow(
            self._custom_trainable_layers_label, self._custom_trainable_layers_spin
        )

        self._custom_backbone_lr_label = QLabel("Backbone LR scale:")
        self._custom_backbone_lr_spin = QDoubleSpinBox()
        self._custom_backbone_lr_spin.setRange(0.001, 1.0)
        self._custom_backbone_lr_spin.setSingleStep(0.01)
        self._custom_backbone_lr_spin.setDecimals(3)
        self._custom_backbone_lr_spin.setValue(0.1)
        self._custom_backbone_lr_spin.setToolTip(
            "LR multiplier applied to unfrozen backbone layers (head LR is full LR)."
        )
        custom_form.addRow(
            self._custom_backbone_lr_label, self._custom_backbone_lr_spin
        )

        self._custom_layerwise_decay_label = QLabel("Layerwise LR decay:")
        self._custom_layerwise_decay_spin = QDoubleSpinBox()
        self._custom_layerwise_decay_spin.setRange(0.1, 1.0)
        self._custom_layerwise_decay_spin.setSingleStep(0.05)
        self._custom_layerwise_decay_spin.setDecimals(2)
        self._custom_layerwise_decay_spin.setValue(0.75)
        self._custom_layerwise_decay_spin.setToolTip(
            "Multiplier applied per stage from head to stem when layerwise LR decay is enabled."
        )
        custom_form.addRow(
            self._custom_layerwise_decay_label, self._custom_layerwise_decay_spin
        )

        self._custom_gradual_unfreeze_interval_label = QLabel(
            "Unfreeze interval (epochs):"
        )
        self._custom_gradual_unfreeze_interval_spin = QSpinBox()
        self._custom_gradual_unfreeze_interval_spin.setRange(1, 100)
        self._custom_gradual_unfreeze_interval_spin.setValue(5)
        self._custom_gradual_unfreeze_interval_spin.setToolTip(
            "For gradual unfreezing, add one more backbone stage every N epochs."
        )
        custom_form.addRow(
            self._custom_gradual_unfreeze_interval_label,
            self._custom_gradual_unfreeze_interval_spin,
        )

        self._custom_input_size_label = QLabel("Input size (px, square):")
        self._custom_input_size_spin = QSpinBox()
        self._custom_input_size_spin.setRange(32, 512)
        self._custom_input_size_spin.setSingleStep(32)
        self._custom_input_size_spin.setValue(224)
        self._custom_input_size_spin.setToolTip(
            "Resize crops to this square size before passing to torchvision backbone."
        )
        custom_form.addRow(self._custom_input_size_label, self._custom_input_size_spin)

        self._custom_general_settings_note = QLabel(
            "Training epochs, batch size, learning rate, and patience always come from the General tab."
        )
        self._custom_general_settings_note.setWordWrap(True)
        self._custom_general_settings_note.setStyleSheet(
            "color:#cfcfcf; font-size:11px;"
        )
        custom_form.addRow("", self._custom_general_settings_note)

        self.tiny_rebalance_combo = QComboBox()
        self.tiny_rebalance_combo.addItem("None", "none")
        self.tiny_rebalance_combo.addItem("Weighted Loss", "weighted_loss")
        self.tiny_rebalance_combo.addItem("Weighted Sampler", "weighted_sampler")
        self.tiny_rebalance_combo.addItem("Weighted Loss + Sampler", "both")
        self.tiny_rebalance_combo.setCurrentIndex(0)
        self.tiny_rebalance_combo.setToolTip(
            "Class imbalance handling for Custom CNN training.\n"
            "Weighted Loss improves minority-class gradients.\n"
            "Weighted Sampler oversamples minority classes per mini-batch."
        )
        custom_form.addRow("Class Rebalancing:", self.tiny_rebalance_combo)

        self.tiny_rebalance_power_spin = QDoubleSpinBox()
        self.tiny_rebalance_power_spin.setRange(0.0, 3.0)
        self.tiny_rebalance_power_spin.setSingleStep(0.1)
        self.tiny_rebalance_power_spin.setValue(1.0)
        self.tiny_rebalance_power_spin.setDecimals(2)
        self.tiny_rebalance_power_spin.setToolTip(
            "Strength of rebalancing (0 disables effect, 1 = inverse-frequency baseline)."
        )
        custom_form.addRow("Rebalance Power:", self.tiny_rebalance_power_spin)

        self._custom_tab_idx = self.tabs.addTab(self.custom_tab, "Custom CNN")

        self._custom_backbone_combo.activated.connect(self._on_custom_backbone_changed)
        self._custom_add_timm_btn.clicked.connect(self._on_add_timm_backbones)
        self.tiny_size_combo.currentIndexChanged.connect(
            lambda _index: self._sync_tiny_preset_controls(self.tiny_size_combo)
        )
        self._tiny_in_custom_size_combo.currentIndexChanged.connect(
            lambda _index: self._sync_tiny_preset_controls(
                self._tiny_in_custom_size_combo
            )
        )
        self._custom_fine_tune_method_combo.currentIndexChanged.connect(
            self._on_custom_backbone_changed
        )
        self._custom_trainable_layers_spin.valueChanged.connect(
            self._on_custom_backbone_changed
        )
        self._on_custom_backbone_changed()
        self._refresh_tiny_preset_summary()

    def _reload_custom_backbone_choices(self, selected: object = None) -> None:
        current = str(selected or self._custom_backbone_combo.currentData() or "")
        self._custom_backbone_combo.blockSignals(True)
        self._custom_backbone_combo.clear()
        for key in get_custom_backbone_choices():
            self._custom_backbone_combo.addItem(custom_backbone_display_name(key), key)
        if current and self._custom_backbone_combo.findData(current) < 0:
            self._custom_backbone_combo.addItem(
                custom_backbone_display_name(current), current
            )
        self._set_combo_value(self._custom_backbone_combo, current or "tinyclassifier")
        self._custom_backbone_combo.blockSignals(False)

    def _on_add_timm_backbones(self) -> None:
        dialog = TimmBackboneBrowserDialog(self)
        if not dialog.exec():
            return
        selected = dialog.selected_backbones()
        if not selected:
            QMessageBox.information(
                self,
                "No Selection",
                "No TIMM backbones were selected.",
            )
            return
        register_user_timm_backbones(selected)
        active = self._custom_backbone_combo.currentData()
        self._reload_custom_backbone_choices(selected=active)

    def _build_space_and_augmentations_tab(self) -> None:
        self.aug_tab = QWidget()
        aug_tab_layout = QVBoxLayout(self.aug_tab)
        aug_tab_layout.setContentsMargins(0, 0, 0, 0)
        self._aug_scroll = QScrollArea()
        self._aug_scroll.setWidgetResizable(True)
        self._aug_scroll.setFrameShape(QFrame.NoFrame)
        aug_scroll_content = QWidget()
        aug_layout = QVBoxLayout(aug_scroll_content)

        aug_group = QGroupBox("Augmentations")
        aug_layout.addWidget(aug_group)
        aug_form = QFormLayout(aug_group)

        self.flip_lr_spin = QDoubleSpinBox()
        self.flip_lr_spin.setRange(0.0, 1.0)
        self.flip_lr_spin.setValue(0.0)
        aug_form.addRow("<b>Horizontal Flip Prob:</b>", self.flip_lr_spin)

        self.flip_ud_spin = QDoubleSpinBox()
        self.flip_ud_spin.setRange(0.0, 1.0)
        self.flip_ud_spin.setValue(0.0)
        aug_form.addRow("<b>Vertical Flip Prob:</b>", self.flip_ud_spin)

        self.brightness_spin = QDoubleSpinBox()
        self.brightness_spin.setRange(0.0, 1.0)
        self.brightness_spin.setSingleStep(0.05)
        self.brightness_spin.setValue(0.0)
        self.brightness_spin.setToolTip(
            "Photometric brightness jitter. Useful for lighting variation while preserving canonical pose."
        )
        aug_form.addRow("<b>Brightness Jitter:</b>", self.brightness_spin)

        self.contrast_spin = QDoubleSpinBox()
        self.contrast_spin.setRange(0.0, 1.0)
        self.contrast_spin.setSingleStep(0.05)
        self.contrast_spin.setValue(0.0)
        self.contrast_spin.setToolTip(
            "Photometric contrast jitter. Preferred over rotation for canonicalized crops."
        )
        aug_form.addRow("<b>Contrast Jitter:</b>", self.contrast_spin)

        self.saturation_spin = QDoubleSpinBox()
        self.saturation_spin.setRange(0.0, 1.0)
        self.saturation_spin.setSingleStep(0.05)
        self.saturation_spin.setValue(0.0)
        self.saturation_spin.setToolTip(
            "Photometric saturation jitter. Useful when color strength varies but hue identity remains informative."
        )
        aug_form.addRow("<b>Saturation Jitter:</b>", self.saturation_spin)

        self.hue_spin = QDoubleSpinBox()
        self.hue_spin.setRange(0.0, 0.5)
        self.hue_spin.setSingleStep(0.01)
        self.hue_spin.setDecimals(2)
        self.hue_spin.setValue(0.0)
        self.hue_spin.setToolTip(
            "Hue jitter fraction. Small values are recommended when color identity may shift across acquisitions."
        )
        aug_form.addRow("<b>Hue Jitter:</b>", self.hue_spin)

        self.monochrome_check = QCheckBox("Force monochrome derived dataset")
        self.monochrome_check.setToolTip(
            "Convert the exported training dataset to grayscale RGB before training. This removes color information across train and validation splits."
        )
        aug_form.addRow("<b>Monochrome Mode:</b>", self.monochrome_check)

        self._exp_group = self._build_expansion_group()
        aug_layout.addWidget(self._exp_group)
        aug_layout.addStretch()
        self._aug_scroll.setWidget(aug_scroll_content)
        aug_tab_layout.addWidget(self._aug_scroll)
        self.tabs.addTab(self.aug_tab, "Space and Augmentations")

    def _build_log_panel(self, layout: QVBoxLayout) -> None:
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFixedHeight(160)
        self.log_view.setStyleSheet(
            "background:#111; color:#ffffff; font-family:monospace; font-size:11px;"
        )
        layout.addWidget(self.log_view)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

    def _build_action_row(self, layout: QVBoxLayout) -> None:
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Training")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.publish_btn = QPushButton("Publish to models/")
        self.publish_btn.setEnabled(False)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.publish_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        total = self._scheme.total_classes if self._scheme else "?"
        name = self._scheme.name if self._scheme else "free-form"
        header = QLabel(
            f"<h2 style='color:#ffffff;margin:0'>Train Classifier</h2>"
            f"<p style='color:#ffffff;margin:4px 0 0 0'>Scheme: <b>{name}</b>"
            f" &nbsp;|&nbsp; Classes: <b>{total}</b>"
            f" &nbsp;|&nbsp; Labeled samples: <b>{self._n_labeled}</b></p>"
        )
        header.setStyleSheet("background: #252526; padding: 10px; border-radius: 4px;")
        layout.addWidget(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_general_tab(), "General")
        self._build_tiny_architecture_tab()
        self._build_custom_cnn_tab()
        self._build_space_and_augmentations_tab()

        layout.addWidget(self.tabs, 1)
        self._build_log_panel(layout)
        self._build_action_row(layout)

        self.cancel_btn.clicked.connect(self._on_cancel)
        self.mode_structure_combo.currentIndexChanged.connect(
            lambda _index: self._on_mode_structure_changed()
        )
        self.mode_family_combo.currentIndexChanged.connect(
            lambda _index: self._sync_mode_combo_from_controls()
        )
        self.mode_combo.currentIndexChanged.connect(self._on_internal_mode_changed)
        self.tiny_rebalance_combo.currentIndexChanged.connect(
            self._on_rebalance_mode_changed
        )
        self._on_internal_mode_changed()
        self._on_rebalance_mode_changed()

        self._exp_group.toggled.connect(
            lambda _checked: self._sync_expansion_constraints()
        )
        self._sync_expansion_constraints()
        self.val_fraction_spin.valueChanged.connect(
            lambda _value: self._refresh_data_summary()
        )
        self.test_fraction_spin.valueChanged.connect(
            lambda _value: self._refresh_data_summary()
        )
        self.split_strategy_combo.currentIndexChanged.connect(
            lambda _value: self._refresh_data_summary()
        )
        self.flip_lr_spin.valueChanged.connect(
            lambda _value: self._refresh_data_summary()
        )
        self.flip_ud_spin.valueChanged.connect(
            lambda _value: self._refresh_data_summary()
        )
        self.brightness_spin.valueChanged.connect(
            lambda _value: self._refresh_data_summary()
        )
        self.contrast_spin.valueChanged.connect(
            lambda _value: self._refresh_data_summary()
        )
        self.saturation_spin.valueChanged.connect(
            lambda _value: self._refresh_data_summary()
        )
        self.hue_spin.valueChanged.connect(lambda _value: self._refresh_data_summary())
        self.monochrome_check.toggled.connect(
            lambda _checked: self._refresh_data_summary()
        )
        self.monochrome_check.toggled.connect(self._sync_sample_preview_controls)
        self._sample_preview_monochrome_toggle.toggled.connect(
            lambda _checked: self._refresh_sample_preview()
        )
        self._sync_sample_preview_controls()
        self._refresh_data_summary()

    def _populate_modes(self):
        self.mode_combo.clear()
        labels = {
            "flat_yolo": "Flat - YOLO-classify",
            "flat_custom": "Flat - Custom CNN",
            "multihead_yolo": "Multi-head - YOLO-classify (one model per factor)",
            "multihead_custom": "Multi-head - Custom CNN (one model per factor)",
            "multihead_custom_shared": "Multi-head - Custom CNN (shared trunk, one .pth)",
        }
        for key in self._resolved_mode_keys():
            self.mode_combo.addItem(labels[key], key)

    def _on_mode_changed(self):
        mode = self.mode_combo.currentData() or ""
        is_yolo = "yolo" in mode
        is_custom = "custom" in mode
        is_multihead = mode.startswith("multihead")

        self.base_model_combo.setVisible(is_yolo)
        if hasattr(self, "_base_model_row_label"):
            self._base_model_row_label.setVisible(is_yolo)

        if hasattr(self, "_initial_model_row"):
            self._initial_model_row.setVisible(not is_multihead)
            self._initial_model_row_label.setVisible(not is_multihead)
            if is_multihead:
                self._initial_model_path_edit.clear()

        expected_suffix = ".pt" if is_yolo else ".pth"
        expected_label = (
            "YOLO weights (.pt)" if is_yolo else "ClassKit checkpoint (.pth)"
        )
        self._initial_model_path_edit.setPlaceholderText(
            f"Optional: choose previous {expected_label} to continue training from it"
        )
        self._initial_model_path_edit.setToolTip(
            "Optional warm start for further training. "
            f"Choose a trainable {expected_suffix} artifact; inference exports like .onnx are not supported."
        )
        self._initial_model_recent_btn.setVisible(
            bool(self._compatible_recent_model_paths())
        )

        if hasattr(self, "_tiny_tab_idx"):
            self.tabs.setTabVisible(self._tiny_tab_idx, False)
            if self.tabs.currentIndex() == self._tiny_tab_idx:
                self.tabs.setCurrentIndex(0)

        if hasattr(self, "_custom_tab_idx"):
            self.tabs.setTabVisible(self._custom_tab_idx, is_custom)
            if not is_custom and self.tabs.currentIndex() == self._custom_tab_idx:
                self.tabs.setCurrentIndex(0)

        _desc = {
            "flat_yolo": (
                "YOLO Classify \u2014 fine-tuned pretrained backbone. Higher accuracy; "
                "GPU strongly recommended. Slower to train."
            ),
            "multihead_yolo": (
                "Multi-head YOLO Classify \u2014 one fine-tuned backbone per factor. "
                "Highest accuracy; GPU required."
            ),
            "flat_custom": (
                "Custom CNN — TinyClassifier plus curated torchvision backbones, with optional TIMM additions. "
                "Supports head-only, full, partial, layerwise-decay, and gradual-unfreezing fine-tuning."
            ),
            "multihead_custom": (
                "Multi-head Custom CNN — one backbone per factor with the same fine-tuning controls as single-head custom training."
            ),
            "multihead_custom_shared": (
                "Multi-head Custom CNN (shared trunk) — single backbone with one classification head per factor, "
                "saved as a single .pth artifact."
            ),
        }
        if hasattr(self, "_mode_desc_label"):
            self._mode_desc_label.setText(_desc.get(mode, ""))

    def _on_custom_backbone_changed(self) -> None:
        backbone = self._custom_backbone_combo.currentData()
        is_tiny = backbone == "tinyclassifier"
        method = str(self._custom_fine_tune_method_combo.currentData() or "head_only")

        for w in (
            self._tiny_in_custom_size_label,
            self._tiny_in_custom_size_combo,
            self._tiny_in_custom_size_summary_label,
            self._tiny_in_custom_width_label,
            self._tiny_in_custom_width_spin,
            self._tiny_in_custom_height_label,
            self._tiny_in_custom_height_spin,
            self._tiny_in_custom_layers_label,
            self._tiny_in_custom_layers_spin,
            self._tiny_in_custom_dim_label,
            self._tiny_in_custom_dim_spin,
            self._tiny_in_custom_dropout_label,
            self._tiny_in_custom_dropout_spin,
        ):
            w.setVisible(is_tiny)

        for w in (
            self._custom_fine_tune_method_combo,
            self._custom_fine_tune_method_label,
            self._custom_trainable_layers_spin,
            self._custom_trainable_layers_label,
            self._custom_input_size_spin,
            self._custom_input_size_label,
            self._custom_layerwise_decay_label,
            self._custom_layerwise_decay_spin,
            self._custom_gradual_unfreeze_interval_label,
            self._custom_gradual_unfreeze_interval_spin,
        ):
            w.setVisible(not is_tiny)

        show_partial = not is_tiny and method == "partial_unfreezing"
        self._custom_trainable_layers_spin.setVisible(show_partial)
        self._custom_trainable_layers_label.setVisible(show_partial)

        show_lr = not is_tiny and method in {"partial_unfreezing", "gradual_unfreezing"}
        self._custom_backbone_lr_spin.setVisible(show_lr)
        self._custom_backbone_lr_label.setVisible(show_lr)
        self._custom_layerwise_decay_label.setVisible(
            not is_tiny and method == "layerwise_lr_decay"
        )
        self._custom_layerwise_decay_spin.setVisible(
            not is_tiny and method == "layerwise_lr_decay"
        )
        self._custom_gradual_unfreeze_interval_label.setVisible(
            not is_tiny and method == "gradual_unfreezing"
        )
        self._custom_gradual_unfreeze_interval_spin.setVisible(
            not is_tiny and method == "gradual_unfreezing"
        )

    def _on_cancel(self):
        if self._worker is not None:
            self._worker.cancel()

    def _on_rebalance_mode_changed(self):
        mode = str(self.tiny_rebalance_combo.currentData() or "none")
        enable_power = mode != "none"
        self.tiny_rebalance_power_spin.setEnabled(enable_power)

    def _current_label_expansion_settings(self) -> dict:
        label_expansion: dict = {}

        def _combo_text(cb: QComboBox) -> str:
            return str(cb.currentData() or cb.currentText() or "").strip()

        if self._exp_group.isChecked():
            lr_map = {
                _combo_text(src): _combo_text(dst)
                for src, dst in self._lr_mapping_rows
                if _combo_text(src)
                and _combo_text(dst)
                and _combo_text(src) != _combo_text(dst)
            }
            if lr_map:
                label_expansion["fliplr"] = lr_map

            ud_map = {
                _combo_text(src): _combo_text(dst)
                for src, dst in self._ud_mapping_rows
                if _combo_text(src)
                and _combo_text(dst)
                and _combo_text(src) != _combo_text(dst)
            }
            if ud_map:
                label_expansion["flipud"] = ud_map

        return label_expansion

    def _planned_training_records_with_splits(self) -> tuple[List[dict], bool]:
        if not self._labeled_label_names:
            return [], False

        records = build_label_expansion_records(
            self._labeled_label_names,
            label_expansion=self._current_label_expansion_settings(),
            groups=(self._group_keys or None),
            known_labels=self._class_choices,
        )
        planned_groups = (
            [record.get("group") for record in records] if self._group_keys else None
        )
        splits, used_group_fallback = build_training_dataset_splits(
            [str(record["label"]) for record in records],
            strategy=self._current_split_strategy(),
            val_fraction=self.val_fraction_spin.value(),
            test_fraction=self.test_fraction_spin.value(),
            groups=planned_groups,
        )
        planned_records = []
        for record, split in zip(records, splits):
            planned_record = dict(record)
            planned_record["split"] = str(split or "train")
            planned_records.append(planned_record)
        return planned_records, bool(used_group_fallback)

    def _current_data_summary(self) -> dict:
        if self._labeled_label_names:
            planned_records, used_group_fallback = (
                self._planned_training_records_with_splits()
            )
            self._used_split_group_fallback = bool(used_group_fallback)
            train_count = sum(
                1 for record in planned_records if record.get("split") == "train"
            )
            val_count = sum(
                1 for record in planned_records if record.get("split") == "val"
            )
            test_count = sum(
                1 for record in planned_records if record.get("split") == "test"
            )
            expansion_count = sum(
                1 for record in planned_records if bool(record.get("is_expanded"))
            )
            return {
                "exact": True,
                "labeled": len(self._labeled_label_names),
                "train": train_count,
                "val": val_count,
                "test": test_count,
                "expansion": expansion_count,
                "exported": len(planned_records),
                "used_group_fallback": bool(used_group_fallback),
            }

        self._used_split_group_fallback = False
        labeled = max(0, int(self._n_labeled))
        val_count = int(round(labeled * float(self.val_fraction_spin.value())))
        test_count = int(round(labeled * float(self.test_fraction_spin.value())))
        if labeled > 1:
            test_count = min(test_count, labeled - 1)
        else:
            test_count = 0
        remaining_after_test = max(0, labeled - test_count)
        if remaining_after_test > 1:
            val_count = min(val_count, remaining_after_test - 1)
        else:
            val_count = 0
        train_count = max(0, labeled - val_count - test_count)
        return {
            "exact": False,
            "labeled": labeled,
            "train": train_count,
            "val": val_count,
            "test": test_count,
            "expansion": 0,
            "exported": labeled,
            "used_group_fallback": False,
        }

    def current_data_summary_text(self) -> str:
        summary = self._current_data_summary()
        strategy_label = (
            "Stratified" if self._current_split_strategy() == "stratified" else "Random"
        )
        lead = (
            f"{strategy_label} export"
            if summary["exact"]
            else f"Estimated {strategy_label.lower()} export"
        )
        lines = [
            (
                f"{lead}: {summary['train']:,} train / {summary['val']:,} val / {summary['test']:,} test "
                f"from {summary['labeled']:,} labeled images."
            )
        ]
        if summary["expansion"] > 0:
            lines.append(
                (
                    f"Label-switching expansion adds {summary['expansion']:,} mirrored copies before splitting, "
                    f"so {summary['exported']:,} files are exported in total."
                )
            )
        else:
            lines.append(f"Exported files: {summary['exported']:,} total.")

        if summary.get("used_group_fallback"):
            lines.append(
                "Related-source grouping was too coarse for the requested holdout, so training falls back to item-level splitting for this export."
            )

        enabled_augments = []
        if self.flip_lr_spin.value() > 0:
            enabled_augments.append(f"LR flip p={self.flip_lr_spin.value():.2f}")
        if self.flip_ud_spin.value() > 0:
            enabled_augments.append(f"UD flip p={self.flip_ud_spin.value():.2f}")
        if self.brightness_spin.value() > 0:
            enabled_augments.append(f"brightness {self.brightness_spin.value():.2f}")
        if self.contrast_spin.value() > 0:
            enabled_augments.append(f"contrast {self.contrast_spin.value():.2f}")
        if self.saturation_spin.value() > 0:
            enabled_augments.append(f"saturation {self.saturation_spin.value():.2f}")
        if self.hue_spin.value() > 0:
            enabled_augments.append(f"hue {self.hue_spin.value():.2f}")
        if self.monochrome_check.isChecked():
            enabled_augments.append("monochrome")

        if enabled_augments:
            lines.append(
                "Train-only stochastic augmentation: "
                + ", ".join(enabled_augments)
                + "."
            )
        else:
            lines.append("Train-only stochastic augmentation: none.")
        return "\n".join(lines)

    def _preview_records_with_splits(self) -> List[dict]:
        planned_records, _used_group_fallback = (
            self._planned_training_records_with_splits()
        )
        if not planned_records or not self._image_paths:
            return []

        records = []
        for record in planned_records:
            source_index = int(record["source_index"])
            if source_index < 0 or source_index >= len(self._image_paths):
                continue
            path = self._image_paths[source_index]
            if Path(path).exists():
                records.append(
                    {
                        "path": Path(path),
                        "label": str(record["label"]),
                        "split": str(record.get("split") or "train"),
                        "expanded": bool(record.get("is_expanded")),
                    }
                )
        return records

    def _current_preview_records(self, max_items: int = 8) -> List[dict]:
        records = self._preview_records_with_splits()
        if len(records) <= max_items:
            return records

        selected = []
        seen_records = set()
        for split_name in ("val", "train"):
            for record in records:
                record_key = (
                    str(record["path"]),
                    str(record["label"]),
                    str(record["split"]),
                    bool(record.get("expanded")),
                )
                if record["split"] != split_name or record_key in seen_records:
                    continue
                selected.append(record)
                seen_records.add(record_key)
                break
            if len(selected) >= max_items:
                return selected[:max_items]

        buckets = {}
        for record in records:
            buckets.setdefault(record["label"], []).append(record)

        for label in sorted(buckets):
            buckets[label] = list(buckets[label])

        while len(selected) < max_items:
            added = False
            for label in sorted(buckets):
                bucket = buckets[label]
                while bucket:
                    bucket_key = (
                        str(bucket[0]["path"]),
                        str(bucket[0]["label"]),
                        str(bucket[0]["split"]),
                        bool(bucket[0].get("expanded")),
                    )
                    if bucket_key not in seen_records:
                        break
                    bucket.pop(0)
                if not bucket:
                    continue
                record = bucket.pop(0)
                selected.append(record)
                seen_records.add(
                    (
                        str(record["path"]),
                        str(record["label"]),
                        str(record["split"]),
                        bool(record.get("expanded")),
                    )
                )
                added = True
                if len(selected) >= max_items:
                    break
            if not added:
                break
        return selected[:max_items]

    def _build_sample_preview_card(self, record: dict) -> QWidget:
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setFixedWidth(150)
        card.setStyleSheet(
            "QFrame { background:#1e1e1e; border:1px solid #3e3e42; border-radius:6px; }"
        )

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(6)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setFixedSize(132, 132)
        image_label.setStyleSheet(
            "background:#111111; border:1px solid #3e3e42; border-radius:4px; color:#cfcfcf;"
        )
        pixmap = self._sample_preview_pixmap(record["path"])
        if pixmap.isNull():
            image_label.setText("Preview\nunavailable")
        else:
            image_label.setPixmap(
                pixmap.scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        card_layout.addWidget(image_label)

        caption = QLabel(
            f"{record['label']}\n{str(record['split']).title()} split\n{record['path'].name}"
        )
        caption.setTextFormat(Qt.PlainText)
        caption.setWordWrap(True)
        caption.setStyleSheet("color:#ffffff; font-size:11px;")
        caption.setToolTip(
            f"Label: {record['label']}\nSplit: {record['split']}\nPath: {record['path']}"
        )
        card_layout.addWidget(caption)

        return card

    def _sample_preview_uses_monochrome(self) -> bool:
        return bool(
            self.monochrome_check.isChecked()
            and self._sample_preview_monochrome_toggle.isChecked()
        )

    def _sample_preview_pixmap(self, path: Path) -> QPixmap:
        pixmap = QPixmap(str(path))
        if pixmap.isNull() or not self._sample_preview_uses_monochrome():
            return pixmap

        image = pixmap.toImage()
        if image.isNull():
            return pixmap
        grayscale = image.convertToFormat(QImage.Format_Grayscale8).convertToFormat(
            QImage.Format_RGB32
        )
        return QPixmap.fromImage(grayscale)

    def _sync_sample_preview_controls(self) -> None:
        monochrome_enabled = self.monochrome_check.isChecked()
        if monochrome_enabled:
            self._sample_preview_monochrome_toggle.setEnabled(True)
            if not self._sample_preview_monochrome_toggle.isChecked():
                self._sample_preview_monochrome_toggle.setChecked(True)
            self._sample_preview_note.setText(
                "Representative labeled source images. With monochrome training enabled, you can preview the grayscale version that will be exported into the training pipeline."
            )
        else:
            if self._sample_preview_monochrome_toggle.isChecked():
                self._sample_preview_monochrome_toggle.setChecked(False)
            self._sample_preview_monochrome_toggle.setEnabled(False)
            self._sample_preview_note.setText(
                "Representative labeled source images before augmentation. Captions show the current label, split, and filename."
            )

    def _refresh_sample_preview(self) -> None:
        if not hasattr(self, "_sample_preview_cards_layout"):
            return

        while self._sample_preview_cards_layout.count():
            item = self._sample_preview_cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        records = self._current_preview_records()
        has_preview_data = bool(records)
        self._sample_preview_group.setVisible(
            has_preview_data or bool(self._image_paths and self._labeled_label_names)
        )

        if not records:
            empty_label = QLabel(
                "No labeled image preview available yet. Add labeled samples to inspect the training inputs here."
            )
            empty_label.setWordWrap(True)
            empty_label.setStyleSheet("color:#cfcfcf; font-size:11px;")
            self._sample_preview_cards_layout.addWidget(empty_label)
            return

        for record in records:
            self._sample_preview_cards_layout.addWidget(
                self._build_sample_preview_card(record)
            )
        self._sample_preview_cards_layout.addStretch()

    def _refresh_data_summary(self) -> None:
        if hasattr(self, "_data_summary_label"):
            self._data_summary_label.setText(self.current_data_summary_text())
        self._refresh_sample_preview()

    def _sync_expansion_constraints(self) -> None:
        expansion_enabled = bool(self._exp_group.isChecked())

        for widget in (
            self.flip_lr_spin,
            self.flip_ud_spin,
        ):
            widget.setEnabled(not expansion_enabled)

        if expansion_enabled:
            self.flip_lr_spin.setValue(0.0)
            self.flip_ud_spin.setValue(0.0)
            self._exp_constraints_label.setText(
                "Label expansion ON: random flips are disabled; hue, saturation, brightness, contrast, and monochrome settings still apply to train samples."
            )
        else:
            self._exp_constraints_label.setText(
                "Label expansion OFF: choose flip, color jitter, and monochrome settings freely."
            )
        self._refresh_data_summary()

    def append_log(self, msg: str):
        if msg:
            self.log_view.appendPlainText(msg)
            self.log_view.ensureCursorVisible()

    def get_settings(self) -> dict:
        label_expansion = self._current_label_expansion_settings()

        expansion_enabled = bool(self._exp_group.isChecked())

        flipud_value = 0.0 if expansion_enabled else self.flip_ud_spin.value()
        fliplr_value = 0.0 if expansion_enabled else self.flip_lr_spin.value()
        hue_value = self.hue_spin.value()
        saturation_value = self.saturation_spin.value()
        brightness_value = self.brightness_spin.value()
        contrast_value = self.contrast_spin.value()
        monochrome_value = self.monochrome_check.isChecked()

        _train_device = str(self.device_combo.currentData() or "cpu")

        _mode = self.mode_combo.currentData() or ""
        _is_custom = _mode in (
            "flat_custom",
            "multihead_custom",
            "multihead_custom_shared",
        )
        fine_tune_method = self._custom_fine_tune_method_combo.currentData()
        custom_backbone = self._custom_backbone_combo.currentData()
        custom_is_tiny = _is_custom and custom_backbone == "tinyclassifier"

        return {
            "mode": self._normalize_mode_key(self.mode_combo.currentData()),
            "device": _train_device,
            "base_model": self.base_model_combo.currentData(),
            "initial_model_path": self._normalize_path_text(
                self._initial_model_path_edit.text()
            ),
            "epochs": self.epochs_spin.value(),
            "batch": self.batch_spin.value(),
            "lr": self.lr_spin.value(),
            "prediction_confidence_threshold": self.prediction_confidence_threshold_spin.value(),
            "split_strategy": self._current_split_strategy(),
            "val_fraction": self.val_fraction_spin.value(),
            "test_fraction": self.test_fraction_spin.value(),
            "patience": self.patience_spin.value(),
            "tiny_preset": (
                self._tiny_in_custom_size_combo.currentData()
                if custom_is_tiny
                else self.tiny_size_combo.currentData()
            ),
            "tiny_layers": (
                self._tiny_in_custom_layers_spin.value()
                if custom_is_tiny
                else self.tiny_layers_spin.value()
            ),
            "tiny_dim": (
                self._tiny_in_custom_dim_spin.value()
                if custom_is_tiny
                else self.tiny_dim_spin.value()
            ),
            "tiny_dropout": (
                self._tiny_in_custom_dropout_spin.value()
                if custom_is_tiny
                else self.tiny_dropout_spin.value()
            ),
            "tiny_width": (
                self._tiny_in_custom_width_spin.value()
                if custom_is_tiny
                else self.tiny_width_spin.value()
            ),
            "tiny_height": (
                self._tiny_in_custom_height_spin.value()
                if custom_is_tiny
                else self.tiny_height_spin.value()
            ),
            "tiny_rebalance_mode": self.tiny_rebalance_combo.currentData(),
            "tiny_rebalance_power": self.tiny_rebalance_power_spin.value(),
            "tiny_label_smoothing": self.tiny_label_smoothing_spin.value(),
            "custom_backbone": custom_backbone,
            "custom_fine_tune_method": fine_tune_method,
            "custom_trainable_layers": self._custom_trainable_layers_spin.value(),
            "custom_backbone_lr_scale": self._custom_backbone_lr_spin.value(),
            "custom_layerwise_lr_decay": self._custom_layerwise_decay_spin.value(),
            "custom_gradual_unfreeze_interval": self._custom_gradual_unfreeze_interval_spin.value(),
            "auto_size_scale_factor": self._auto_size_scale_spin.value(),
            "custom_input_size": self._custom_input_size_spin.value(),
            "flipud": flipud_value,
            "fliplr": fliplr_value,
            "hue": hue_value,
            "saturation": saturation_value,
            "brightness": brightness_value,
            "contrast": contrast_value,
            "monochrome": monochrome_value,
            "label_expansion": label_expansion,
        }
