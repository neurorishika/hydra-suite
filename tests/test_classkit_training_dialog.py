from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hydra_suite.classkit.config.schemas import Factor, LabelingScheme

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QtGui = pytest.importorskip("PySide6.QtGui")
QApplication = QtWidgets.QApplication
QScrollArea = QtWidgets.QScrollArea
QColor = QtGui.QColor
QImage = QtGui.QImage
QLabel = QtWidgets.QLabel

ClassKitTrainingDialog = pytest.importorskip(
    "hydra_suite.classkit.gui.dialogs.training"
).ClassKitTrainingDialog


@pytest.fixture()
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture(autouse=True)
def cleanup_qt_widgets(qapp):
    yield
    for widget in list(qapp.topLevelWidgets()):
        widget.close()
        widget.deleteLater()
    qapp.processEvents()
    gc.collect()


def _write_test_image(path: Path, color_name: str) -> None:
    image = QImage(32, 24, QImage.Format_RGB32)
    image.fill(QColor(color_name))
    assert image.save(str(path)) is True


def _preview_card_image_label(dialog: ClassKitTrainingDialog) -> QLabel:
    for index in range(dialog._sample_preview_cards_layout.count()):
        card = dialog._sample_preview_cards_layout.itemAt(index).widget()
        if card is None:
            continue
        layout = card.layout()
        if layout is None or layout.count() == 0:
            continue
        image_label = layout.itemAt(0).widget()
        if isinstance(image_label, QLabel) and image_label.pixmap() is not None:
            return image_label
    raise AssertionError("No preview image label found")


def test_training_dialog_auto_sizes_from_average_image_dimensions(qapp) -> None:
    dialog = ClassKitTrainingDialog(
        n_labeled=8,
        class_choices=["a", "b"],
        average_image_size=(101.0, 67.0),
    )

    assert dialog.tiny_width_spin.value() == 96
    assert dialog.tiny_height_spin.value() == 64
    assert dialog._tiny_in_custom_width_spin.value() == 96
    assert dialog._tiny_in_custom_height_spin.value() == 64
    assert dialog._custom_input_size_spin.value() == 96


def test_training_dialog_augmentation_defaults_start_disabled(qapp) -> None:
    dialog = ClassKitTrainingDialog(n_labeled=8, class_choices=["a", "b"])

    settings = dialog.get_settings()

    assert dialog.flip_ud_spin.value() == pytest.approx(0.0)
    assert dialog.flip_lr_spin.value() == pytest.approx(0.0)
    assert dialog.hue_spin.value() == pytest.approx(0.0)
    assert dialog.saturation_spin.value() == pytest.approx(0.0)
    assert dialog.brightness_spin.value() == pytest.approx(0.0)
    assert dialog.contrast_spin.value() == pytest.approx(0.0)
    assert dialog.monochrome_check.isChecked() is False
    assert settings["flipud"] == pytest.approx(0.0)
    assert settings["fliplr"] == pytest.approx(0.0)
    assert settings["hue"] == pytest.approx(0.0)
    assert settings["saturation"] == pytest.approx(0.0)
    assert settings["brightness"] == pytest.approx(0.0)
    assert settings["contrast"] == pytest.approx(0.0)
    assert settings["monochrome"] is False
    assert settings["split_strategy"] == "stratified"


def test_training_dialog_restores_initial_settings(qapp) -> None:
    dialog = ClassKitTrainingDialog(
        n_labeled=8,
        class_choices=["a", "b"],
        average_image_size=(101.0, 67.0),
        initial_settings={
            "mode": "flat_custom",
            "device": "cpu",
            "compute_runtime": "cpu",
            "custom_backbone": "resnet18",
            "auto_size_scale_factor": 1.25,
            "custom_input_size": 192,
            "tiny_preset": "large",
            "tiny_width": 160,
            "tiny_height": 96,
            "epochs": 30,
            "batch": 16,
            "lr": 0.002,
            "test_fraction": 0.1,
            "patience": 7,
            "split_strategy": "random",
            "custom_fine_tune_method": "layerwise_lr_decay",
            "custom_layerwise_lr_decay": 0.6,
            "hue": 0.04,
            "saturation": 0.35,
            "brightness": 0.2,
            "contrast": 0.15,
            "monochrome": True,
            "initial_model_path": "/tmp/previous_model.pth",
        },
    )

    assert dialog.mode_combo.currentData() == "flat_custom"
    assert dialog.mode_structure_combo.currentData() == "flat"
    assert dialog.mode_family_combo.currentData() == "custom"
    assert dialog._custom_backbone_combo.currentData() == "resnet18"
    assert dialog._auto_size_scale_spin.value() == pytest.approx(1.25)
    assert dialog._custom_input_size_spin.value() == 192
    assert dialog._custom_fine_tune_method_combo.currentData() == "layerwise_lr_decay"
    assert dialog._custom_layerwise_decay_spin.value() == pytest.approx(0.6)
    assert dialog.hue_spin.value() == pytest.approx(0.04)
    assert dialog.saturation_spin.value() == pytest.approx(0.35)
    assert dialog.brightness_spin.value() == pytest.approx(0.2)
    assert dialog.contrast_spin.value() == pytest.approx(0.15)
    assert dialog.monochrome_check.isChecked() is True
    assert dialog.tiny_size_combo.currentData() == "large"
    assert dialog._tiny_in_custom_size_combo.currentData() == "large"
    assert dialog.tiny_width_spin.value() == 160
    assert dialog.tiny_height_spin.value() == 96
    assert dialog.epochs_spin.value() == 30
    assert dialog.batch_spin.value() == 16
    assert dialog.lr_spin.value() == pytest.approx(0.002)
    assert dialog.test_fraction_spin.value() == pytest.approx(0.1)
    assert dialog.patience_spin.value() == 7
    assert dialog.split_strategy_combo.currentData() == "random"
    assert dialog.get_settings()["initial_model_path"] == "/tmp/previous_model.pth"


def test_training_dialog_tiny_preset_is_mirrored_between_modes(qapp) -> None:
    dialog = ClassKitTrainingDialog(n_labeled=8, class_choices=["a", "b"])

    dialog.tiny_size_combo.setCurrentIndex(dialog.tiny_size_combo.findData("large"))

    assert dialog._tiny_in_custom_size_combo.currentData() == "large"
    assert "Tiny-L" in dialog._tiny_size_summary_label.text()
    assert dialog.get_settings()["tiny_preset"] == "large"

    dialog.mode_family_combo.setCurrentIndex(
        dialog.mode_family_combo.findData("custom")
    )
    dialog._custom_backbone_combo.setCurrentIndex(
        dialog._custom_backbone_combo.findData("tinyclassifier")
    )

    dialog._tiny_in_custom_size_combo.setCurrentIndex(
        dialog._tiny_in_custom_size_combo.findData("small")
    )

    settings = dialog.get_settings()

    assert dialog.tiny_size_combo.currentData() == "small"
    assert settings["tiny_preset"] == "small"


def test_training_dialog_custom_mode_uses_general_hyperparams(qapp) -> None:
    dialog = ClassKitTrainingDialog(
        n_labeled=8,
        class_choices=["a", "b"],
        initial_settings={
            "mode": "flat_custom",
            "custom_backbone": "resnet18",
        },
    )

    dialog.epochs_spin.setValue(17)
    dialog.batch_spin.setValue(24)
    dialog.lr_spin.setValue(0.003)
    dialog.patience_spin.setValue(6)

    settings = dialog.get_settings()

    assert settings["epochs"] == 17
    assert settings["batch"] == 24
    assert settings["lr"] == pytest.approx(0.003)
    assert settings["patience"] == 6


def test_training_dialog_auto_size_helper_applies_rescale_factor(qapp) -> None:
    dialog = ClassKitTrainingDialog(n_labeled=8, class_choices=["a", "b"])
    dialog._image_paths = [Path("image_a.png")]
    dialog._auto_size_btn.setEnabled(True)
    dialog._auto_size_scale_spin.setValue(1.5)
    dialog._sample_image_dimensions = lambda: [(100.0, 50.0)]

    dialog._auto_set_sizes_from_images()

    assert dialog.tiny_width_spin.value() == 160
    assert dialog.tiny_height_spin.value() == 64
    assert dialog._tiny_in_custom_width_spin.value() == 160
    assert dialog._tiny_in_custom_height_spin.value() == 64
    assert dialog._custom_input_size_spin.value() == 160
    assert dialog.get_settings()["auto_size_scale_factor"] == pytest.approx(1.5)


def test_training_dialog_split_strategy_can_be_switched_to_random(qapp) -> None:
    dialog = ClassKitTrainingDialog(n_labeled=8, class_choices=["a", "b"])

    dialog.split_strategy_combo.setCurrentIndex(
        dialog.split_strategy_combo.findData("random")
    )

    settings = dialog.get_settings()

    assert settings["split_strategy"] == "random"


def test_training_dialog_prefers_onnx_coreml_for_mps_inference(
    qapp, monkeypatch
) -> None:
    import hydra_suite.runtime.compute_runtime as compute_runtime

    monkeypatch.setattr(
        compute_runtime,
        "supported_runtimes_for_pipeline",
        lambda _pipeline: ["cpu", "mps", "onnx_coreml", "onnx_cpu"],
    )

    dialog = ClassKitTrainingDialog(
        n_labeled=8,
        class_choices=["a", "b"],
        initial_settings={"device": "mps"},
    )

    assert dialog.compute_runtime_combo.currentData() == "onnx_coreml"


def test_training_dialog_data_summary_reflects_stratified_split_and_expansion(
    qapp,
) -> None:
    dialog = ClassKitTrainingDialog(
        n_labeled=10,
        class_choices=["left", "right"],
        labeled_label_names=["left"] * 6 + ["right"] * 4,
    )

    dialog._exp_group.setChecked(True)
    dialog._add_lr_row("left", "right")

    summary = dialog.current_data_summary_text()

    assert "13 train / 3 val / 0 test" in summary
    assert "adds 6 mirrored copies before splitting" in summary
    assert "16 files are exported" in summary


def test_training_dialog_grouped_preview_includes_test_split(qapp) -> None:
    labels = ["left", "left", "left", "left", "right", "right", "right", "right"]
    dialog = ClassKitTrainingDialog(
        n_labeled=len(labels),
        class_choices=["left", "right"],
        labeled_label_names=labels,
        group_keys=["a", "a", "b", "b", "c", "c", "d", "d"],
    )

    dialog.val_fraction_spin.setValue(0.25)
    dialog.test_fraction_spin.setValue(0.25)

    summary = dialog.current_data_summary_text()
    settings = dialog.get_settings()

    assert "4 train / 2 val / 2 test" in summary
    assert settings["test_fraction"] == pytest.approx(0.25)


def test_training_dialog_grouped_random_preview_preserves_train_when_only_one_group(
    qapp,
) -> None:
    labels = ["left"] * 10
    dialog = ClassKitTrainingDialog(
        n_labeled=len(labels),
        class_choices=["left"],
        labeled_label_names=labels,
        group_keys=["session-a"] * len(labels),
    )

    dialog.split_strategy_combo.setCurrentIndex(
        dialog.split_strategy_combo.findData("random")
    )
    dialog.val_fraction_spin.setValue(0.2)

    summary = dialog.current_data_summary_text()

    assert "8 train / 2 val / 0 test" in summary
    assert "item-level splitting" in summary


def test_training_dialog_grouped_stratified_preview_preserves_train_when_only_one_group(
    qapp,
) -> None:
    labels = ["left"] * 10
    dialog = ClassKitTrainingDialog(
        n_labeled=len(labels),
        class_choices=["left"],
        labeled_label_names=labels,
        group_keys=["session-a"] * len(labels),
    )

    dialog.val_fraction_spin.setValue(0.2)

    summary = dialog.current_data_summary_text()

    assert "8 train / 2 val / 0 test" in summary
    assert "item-level splitting" in summary


def test_training_dialog_shows_labeled_sample_preview(qapp, tmp_path: Path) -> None:
    image_paths = []
    for idx, color_name in enumerate(
        ["red", "green", "blue", "yellow", "cyan", "magenta"]
    ):
        image_path = tmp_path / f"sample_{idx}.png"
        _write_test_image(image_path, color_name)
        image_paths.append(image_path)

    labels = ["left", "left", "left", "right", "right", "right"]
    dialog = ClassKitTrainingDialog(
        n_labeled=len(labels),
        class_choices=["left", "right"],
        labeled_label_names=labels,
        image_paths=image_paths,
    )

    records = dialog._current_preview_records()
    preview_widgets = [
        dialog._sample_preview_cards_layout.itemAt(i).widget()
        for i in range(dialog._sample_preview_cards_layout.count())
        if dialog._sample_preview_cards_layout.itemAt(i).widget() is not None
    ]

    assert not dialog._sample_preview_group.isHidden()
    assert len(records) == len(preview_widgets)
    assert {record["label"] for record in records} == {"left", "right"}
    assert any(record["split"] == "val" for record in records)


def test_training_dialog_monochrome_preview_toggle_matches_training_mode(
    qapp, tmp_path: Path
) -> None:
    image_path = tmp_path / "sample.png"
    _write_test_image(image_path, "red")

    dialog = ClassKitTrainingDialog(
        n_labeled=2,
        class_choices=["left", "right"],
        labeled_label_names=["left", "right"],
        image_paths=[image_path, image_path],
    )

    assert dialog._sample_preview_monochrome_toggle.isEnabled() is False

    dialog.monochrome_check.setChecked(True)

    assert dialog._sample_preview_monochrome_toggle.isEnabled() is True
    assert dialog._sample_preview_monochrome_toggle.isChecked() is True

    image_label = _preview_card_image_label(dialog)
    pixmap = image_label.pixmap()
    color = pixmap.toImage().pixelColor(pixmap.width() // 2, pixmap.height() // 2)

    assert color.red() == color.green() == color.blue()


def test_training_dialog_preview_toggle_can_show_original_color_when_monochrome_enabled(
    qapp, tmp_path: Path
) -> None:
    image_path = tmp_path / "sample.png"
    _write_test_image(image_path, "red")

    dialog = ClassKitTrainingDialog(
        n_labeled=2,
        class_choices=["left", "right"],
        labeled_label_names=["left", "right"],
        image_paths=[image_path, image_path],
    )

    dialog.monochrome_check.setChecked(True)
    dialog._sample_preview_monochrome_toggle.setChecked(False)

    image_label = _preview_card_image_label(dialog)
    pixmap = image_label.pixmap()
    color = pixmap.toImage().pixelColor(pixmap.width() // 2, pixmap.height() // 2)

    assert color.red() > color.green()
    assert color.red() > color.blue()


def test_training_dialog_label_expansion_keeps_photometric_jitter(qapp) -> None:
    dialog = ClassKitTrainingDialog(
        n_labeled=6,
        class_choices=["left", "right"],
        labeled_label_names=["left", "right"] * 3,
    )

    dialog.hue_spin.setValue(0.03)
    dialog.saturation_spin.setValue(0.25)
    dialog.brightness_spin.setValue(0.2)
    dialog.contrast_spin.setValue(0.15)
    dialog.monochrome_check.setChecked(True)
    dialog.flip_lr_spin.setValue(0.4)
    dialog._exp_group.setChecked(True)
    dialog._add_lr_row("left", "right")

    settings = dialog.get_settings()

    assert settings["fliplr"] == pytest.approx(0.0)
    assert settings["flipud"] == pytest.approx(0.0)
    assert settings["hue"] == pytest.approx(0.03)
    assert settings["saturation"] == pytest.approx(0.25)
    assert settings["brightness"] == pytest.approx(0.2)
    assert settings["contrast"] == pytest.approx(0.15)
    assert settings["monochrome"] is True
    assert settings["label_expansion"] == {"fliplr": {"left": "right"}}


def test_training_dialog_summary_mentions_color_jitter_and_monochrome(qapp) -> None:
    dialog = ClassKitTrainingDialog(n_labeled=8, class_choices=["a", "b"])

    dialog.hue_spin.setValue(0.05)
    dialog.saturation_spin.setValue(0.3)
    dialog.monochrome_check.setChecked(True)

    summary = dialog.current_data_summary_text()

    assert "hue 0.05" in summary
    assert "saturation 0.30" in summary
    assert "monochrome" in summary


def test_training_dialog_exposes_custom_and_yolo_modes_only(qapp) -> None:
    dialog = ClassKitTrainingDialog(n_labeled=8, class_choices=["a", "b"])

    modes = [dialog.mode_combo.itemData(i) for i in range(dialog.mode_combo.count())]
    structures = [
        dialog.mode_structure_combo.itemData(i)
        for i in range(dialog.mode_structure_combo.count())
    ]
    families = [
        dialog.mode_family_combo.itemData(i)
        for i in range(dialog.mode_family_combo.count())
    ]

    assert modes == ["flat_custom", "flat_yolo"]
    assert structures == ["flat"]
    assert families == ["custom", "yolo"]


def test_training_dialog_exposes_multihead_modes_for_multifactor_scheme(qapp) -> None:
    dialog = ClassKitTrainingDialog(
        scheme=LabelingScheme(
            name="tags",
            factors=[
                Factor(name="tag_1", labels=["red", "blue"]),
                Factor(name="tag_2", labels=["left", "right"]),
            ],
            training_modes=[
                "flat_tiny",
                "flat_yolo",
                "multihead_yolo",
                "multihead_tiny",
            ],
        ),
        n_labeled=8,
        class_choices=["red|left", "blue|right"],
    )

    modes = [dialog.mode_combo.itemData(i) for i in range(dialog.mode_combo.count())]
    structures = [
        dialog.mode_structure_combo.itemData(i)
        for i in range(dialog.mode_structure_combo.count())
    ]

    assert modes == [
        "flat_custom",
        "flat_yolo",
        "multihead_yolo",
        "multihead_custom",
    ]
    assert structures == ["flat", "multihead"]


def test_training_dialog_falls_back_to_multihead_modes_for_legacy_scheme(qapp) -> None:
    dialog = ClassKitTrainingDialog(
        scheme=LabelingScheme(
            name="legacy_tags",
            factors=[
                Factor(name="tag_1", labels=["red", "blue"]),
                Factor(name="tag_2", labels=["left", "right"]),
            ],
            training_modes=[],
        ),
        n_labeled=8,
        class_choices=["red|left", "blue|right"],
    )

    modes = [dialog.mode_combo.itemData(i) for i in range(dialog.mode_combo.count())]

    assert modes == [
        "flat_custom",
        "flat_yolo",
        "multihead_yolo",
        "multihead_custom",
    ]


def test_training_dialog_split_mode_controls_sync_internal_mode(qapp) -> None:
    dialog = ClassKitTrainingDialog(
        scheme=LabelingScheme(
            name="tags",
            factors=[
                Factor(name="tag_1", labels=["red", "blue"]),
                Factor(name="tag_2", labels=["left", "right"]),
            ],
            training_modes=[
                "flat_custom",
                "flat_yolo",
                "multihead_yolo",
                "multihead_custom",
            ],
        ),
        n_labeled=8,
        class_choices=["red|left", "blue|right"],
    )

    dialog.mode_structure_combo.setCurrentIndex(
        dialog.mode_structure_combo.findData("multihead")
    )
    dialog.mode_family_combo.setCurrentIndex(dialog.mode_family_combo.findData("yolo"))

    assert dialog.mode_combo.currentData() == "multihead_yolo"
    assert dialog.get_settings()["mode"] == "multihead_yolo"


def test_training_dialog_custom_tab_uses_scroll_area(qapp) -> None:
    dialog = ClassKitTrainingDialog(n_labeled=8, class_choices=["a", "b"])

    custom_tab = dialog.tabs.widget(dialog._custom_tab_idx)

    assert isinstance(custom_tab, QScrollArea)
    assert custom_tab.widgetResizable()


def test_training_dialog_class_rebalancing_defaults_to_none(qapp) -> None:
    dialog = ClassKitTrainingDialog(n_labeled=8, class_choices=["a", "b"])

    settings = dialog.get_settings()

    assert settings["tiny_rebalance_mode"] == "none"
    assert not dialog.tiny_rebalance_power_spin.isEnabled()


def test_training_dialog_filters_recent_model_choices_by_mode(qapp) -> None:
    recent_yolo = str(Path("/tmp/run_a/weights/best.pt").resolve())
    recent_custom = str(Path("/tmp/run_b/weights/best.pth").resolve())
    dialog = ClassKitTrainingDialog(
        n_labeled=8,
        class_choices=["a", "b"],
        recent_model_paths=[
            "/tmp/run_a/weights/best.pt",
            "/tmp/run_b/weights/best.pth",
        ],
    )

    dialog.mode_structure_combo.setCurrentIndex(
        dialog.mode_structure_combo.findData("flat")
    )
    dialog.mode_family_combo.setCurrentIndex(
        dialog.mode_family_combo.findData("custom")
    )
    assert dialog._compatible_recent_model_paths() == [recent_custom]

    dialog.mode_family_combo.setCurrentIndex(dialog.mode_family_combo.findData("yolo"))
    assert dialog._compatible_recent_model_paths() == [recent_yolo]
