from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from hydra_suite.trackerkit.gui.main_window import MainWindow
from hydra_suite.trackerkit.gui.orchestrators.config import ConfigOrchestrator


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_main_window(
    monkeypatch: pytest.MonkeyPatch,
    advanced_config: dict[str, object] | None = None,
) -> MainWindow:
    monkeypatch.setattr(MainWindow, "_save_advanced_config", lambda self: None)
    monkeypatch.setattr(
        MainWindow,
        "_load_advanced_config",
        lambda self: dict(advanced_config or {}),
    )
    return MainWindow()


def _select_first_model_with_suffix(combo, suffix: str) -> str:
    for index in range(combo.count()):
        item_data = combo.itemData(index)
        if isinstance(item_data, str) and item_data.endswith(suffix):
            combo.setCurrentIndex(index)
            return item_data
    raise AssertionError(f"No model ending with {suffix!r} was available in the combo")


def _select_first_model_with_suffixes(combo, suffixes: tuple[str, ...]) -> str:
    for suffix in suffixes:
        try:
            return _select_first_model_with_suffix(combo, suffix)
        except AssertionError:
            continue
    raise AssertionError(
        f"No model ending with {suffixes!r} was available in the combo"
    )


def _select_first_nonempty_model(combo) -> str:
    for index in range(combo.count()):
        item_data = combo.itemData(index)
        if item_data and item_data not in ("__add_new__", "__none__"):
            combo.setCurrentIndex(index)
            return str(item_data)
    raise AssertionError("No configured model entry was available in the combo")


def _seed_trackerkit_model_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    data_dir = tmp_path / "hydra-data"
    monkeypatch.setenv("HYDRA_DATA_DIR", str(data_dir))
    models_root = data_dir / "models"
    registry: dict[str, object] = {}

    def add_file(rel_path: str, metadata: dict[str, object] | None = None) -> None:
        path = models_root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stub model", encoding="utf-8")
        if metadata is not None:
            registry[rel_path] = metadata

    def add_dir(rel_path: str) -> None:
        path = models_root / rel_path
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")

    add_file(
        "obb/direct_remove.pt",
        {
            "task_family": "obb",
            "usage_role": "obb_direct",
            "size": "26s",
            "species": "ant",
            "model_info": "direct_remove",
        },
    )
    add_file(
        "obb/direct_keep.pt",
        {
            "task_family": "obb",
            "usage_role": "obb_direct",
            "size": "26s",
            "species": "ant",
            "model_info": "direct_keep",
        },
    )
    add_file(
        "detection/seq_detect_remove.pt",
        {
            "task_family": "detect",
            "usage_role": "seq_detect",
            "size": "26s",
            "species": "ant",
            "model_info": "seq_detect_remove",
        },
    )
    add_file(
        "detection/seq_detect_keep.pt",
        {
            "task_family": "detect",
            "usage_role": "seq_detect",
            "size": "26s",
            "species": "ant",
            "model_info": "seq_detect_keep",
        },
    )
    add_file(
        "obb/cropped/seq_crop_remove.pt",
        {
            "task_family": "obb",
            "usage_role": "seq_crop_obb",
            "size": "26s",
            "species": "ant",
            "model_info": "seq_crop_remove",
        },
    )
    add_file(
        "obb/cropped/seq_crop_keep.pt",
        {
            "task_family": "obb",
            "usage_role": "seq_crop_obb",
            "size": "26s",
            "species": "ant",
            "model_info": "seq_crop_keep",
        },
    )
    add_file(
        "classification/orientation/YOLO/headtail_remove.pt",
        {
            "task_family": "classify",
            "usage_role": "headtail",
            "size": "26s",
            "species": "ant",
            "model_info": "headtail_remove",
        },
    )
    add_file(
        "classification/orientation/YOLO/headtail_keep.pt",
        {
            "task_family": "classify",
            "usage_role": "headtail",
            "size": "26s",
            "species": "ant",
            "model_info": "headtail_keep",
        },
    )
    add_file(
        "classification/identity/cnn_remove.pt",
        {
            "task_family": "classify",
            "usage_role": "cnn_identity",
            "arch": "tinyclassifier",
            "num_classes": 2,
            "class_names": ["worker", "queen"],
            "factor_names": [],
            "input_size": [224, 224],
            "classification_label": "colony",
            "species": "ant",
            "model_info": "cnn_remove",
        },
    )
    add_file(
        "classification/identity/cnn_keep.pt",
        {
            "task_family": "classify",
            "usage_role": "cnn_identity",
            "arch": "tinyclassifier",
            "num_classes": 2,
            "class_names": ["worker", "queen"],
            "factor_names": [],
            "input_size": [224, 224],
            "classification_label": "colony",
            "species": "ant",
            "model_info": "cnn_keep",
        },
    )
    add_file("pose/YOLO/pose_remove.pt")
    add_file("pose/YOLO/pose_keep.pt")
    add_dir("pose/SLEAP/sleap_remove")
    add_dir("pose/SLEAP/sleap_keep")

    registry_path = models_root / "model_registry.json"
    registry_path.write_text(
        json.dumps({"schema_version": 2, "entries": registry}, indent=2),
        encoding="utf-8",
    )
    return models_root


def _load_seeded_model_registry(models_root: Path) -> dict[str, object]:
    payload = json.loads(
        (models_root / "model_registry.json").read_text(encoding="utf-8")
    )
    return dict(payload.get("entries") or {})


def test_headtail_model_roundtrip_preserves_selection(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    window = _make_main_window(monkeypatch)
    window._identity_panel.g_headtail.setChecked(True)
    window._identity_panel._refresh_yolo_headtail_model_combo()

    selected_model = _select_first_model_with_suffix(
        window._identity_panel.combo_yolo_headtail_model,
        ".pth",
    )
    assert selected_model

    config_path = tmp_path / "headtail_roundtrip.json"
    assert window.save_config(preset_mode=True, preset_path=str(config_path))
    saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_cfg["enable_headtail_orientation"] is True
    assert "yolo_headtail_model_type" not in saved_cfg
    window.close()

    reloaded_window = _make_main_window(monkeypatch)
    reloaded_window._load_config_from_file(str(config_path), preset_mode=True)

    assert reloaded_window._identity_panel.g_headtail.isChecked() is True
    assert (
        reloaded_window._identity_panel._get_selected_yolo_headtail_model_path()
        == selected_model
    )
    reloaded_window.close()


def test_headtail_toggle_roundtrip_preserves_selection_but_disables_runtime_use(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    window = _make_main_window(monkeypatch)
    selected_model = _select_first_model_with_suffixes(
        window._identity_panel.combo_yolo_headtail_model,
        (".pt", ".pth"),
    )
    window._identity_panel.g_headtail.setChecked(False)

    config_path = tmp_path / "headtail_toggle_roundtrip.json"
    assert window.save_config(preset_mode=True, preset_path=str(config_path))
    saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))

    assert saved_cfg["enable_headtail_orientation"] is False
    assert saved_cfg["yolo_headtail_model_path"]
    assert window.get_parameters_dict()["YOLO_HEADTAIL_MODEL_PATH"] == ""
    window.close()

    reloaded_window = _make_main_window(monkeypatch)
    reloaded_window._load_config_from_file(str(config_path), preset_mode=True)

    assert reloaded_window._identity_panel.g_headtail.isChecked() is False
    assert (
        reloaded_window._identity_panel._get_configured_yolo_headtail_model_path()
        == selected_model
    )
    assert reloaded_window.get_parameters_dict()["YOLO_HEADTAIL_MODEL_PATH"] == ""
    reloaded_window.close()


def test_headtail_batch_size_roundtrip_preserves_runtime_parameter(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    window = _make_main_window(monkeypatch)
    window._identity_panel.g_headtail.setChecked(True)
    window._identity_panel.spin_headtail_batch.setValue(48)

    config_path = tmp_path / "headtail_batch_roundtrip.json"
    assert window.save_config(preset_mode=True, preset_path=str(config_path))
    saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))

    assert saved_cfg["headtail_batch_size"] == 48
    assert window.get_parameters_dict()["HEADTAIL_BATCH_SIZE"] == 48
    window.close()

    reloaded_window = _make_main_window(monkeypatch)
    reloaded_window._load_config_from_file(str(config_path), preset_mode=True)

    assert reloaded_window._identity_panel.spin_headtail_batch.value() == 48
    assert reloaded_window.get_parameters_dict()["HEADTAIL_BATCH_SIZE"] == 48
    reloaded_window.close()


def test_headtail_detection_confidence_roundtrip_preserves_runtime_parameter(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    window = _make_main_window(monkeypatch)
    window._identity_panel.g_headtail.setChecked(True)
    window._identity_panel.spin_yolo_headtail_detect_conf.setValue(0.72)

    config_path = tmp_path / "headtail_detect_conf_roundtrip.json"
    assert window.save_config(preset_mode=True, preset_path=str(config_path))
    saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))

    assert saved_cfg["yolo_headtail_detect_conf_threshold"] == pytest.approx(0.72)
    assert window.get_parameters_dict()[
        "YOLO_HEADTAIL_DETECT_CONF_THRESHOLD"
    ] == pytest.approx(0.72)
    window.close()

    reloaded_window = _make_main_window(monkeypatch)
    reloaded_window._load_config_from_file(str(config_path), preset_mode=True)

    assert (
        reloaded_window._identity_panel.spin_yolo_headtail_detect_conf.value()
        == pytest.approx(0.72)
    )
    assert reloaded_window.get_parameters_dict()[
        "YOLO_HEADTAIL_DETECT_CONF_THRESHOLD"
    ] == pytest.approx(0.72)
    reloaded_window.close()


def test_headtail_combo_includes_registry_annotated_classkit_artifact(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
    tiny_flat_headtail: Path,
) -> None:
    from hydra_suite.training.model_publish import import_classifier_artifact

    models_root = _seed_trackerkit_model_repository(tmp_path, monkeypatch)
    rel_path = "tiny-classify/scheme/discovered_headtail.pth"
    artifact_path = models_root / rel_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(tiny_flat_headtail), str(artifact_path))

    registered_path = import_classifier_artifact(
        source_path=artifact_path,
        usage_role="head_tail",
        species="ant",
        description="annotated external headtail",
    )

    window = _make_main_window(monkeypatch)
    window._identity_panel.g_headtail.setChecked(True)
    window._identity_panel._refresh_yolo_headtail_model_combo(
        preferred_model_path=registered_path
    )

    combo = window._identity_panel.combo_yolo_headtail_model
    assert combo.findData(registered_path) >= 0
    assert (
        window._identity_panel._get_configured_yolo_headtail_model_path()
        == registered_path
    )
    window.close()


def test_headtail_selector_ignores_none_and_registered_alias_entries(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    _seed_trackerkit_model_repository(tmp_path, monkeypatch)
    window = _make_main_window(monkeypatch)
    window._identity_panel.g_headtail.setChecked(True)
    window._identity_panel._refresh_yolo_headtail_model_combo()

    combo = window._identity_panel.combo_yolo_headtail_model
    none_index = combo.findData("__none__")
    registered_index = combo.findData(
        "classification/orientation/YOLO/headtail_keep.pt"
    )
    assert none_index >= 0
    assert registered_index >= 0

    add_calls: list[object] = []
    annotate_calls: list[object] = []
    sync_calls: list[object] = []

    monkeypatch.setattr(
        window,
        "_handle_add_new_headtail_model",
        lambda: add_calls.append("add"),
    )
    monkeypatch.setattr(
        window,
        "_annotate_discovered_headtail_model",
        lambda rel_path: annotate_calls.append(rel_path),
    )
    monkeypatch.setattr(
        window,
        "_sync_individual_analysis_mode_ui",
        lambda: sync_calls.append("sync"),
    )

    window.on_yolo_headtail_model_changed(none_index)
    window.on_yolo_headtail_model_changed(registered_index)

    assert add_calls == []
    assert annotate_calls == []
    assert sync_calls == ["sync", "sync"]
    window.close()


def test_refinekit_prompt_toggle_roundtrip_and_batch_mode_clears_it(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    window = _make_main_window(monkeypatch)
    window._postprocess_panel.chk_prompt_open_refinekit.setChecked(True)

    config_path = tmp_path / "refinekit_prompt_roundtrip.json"
    assert window.save_config(preset_mode=True, preset_path=str(config_path))
    saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))

    assert saved_cfg["prompt_open_refinekit_on_tracking_complete"] is True
    assert window._postprocess_panel.g_refinekit.isHidden() is False

    window._setup_panel.g_batch.setChecked(True)
    assert window._postprocess_panel.g_refinekit.isHidden() is True
    assert window._postprocess_panel.chk_prompt_open_refinekit.isChecked() is False

    window._setup_panel.g_batch.setChecked(False)
    assert window._postprocess_panel.g_refinekit.isHidden() is False
    assert window._postprocess_panel.chk_prompt_open_refinekit.isChecked() is False
    window.close()

    reloaded_window = _make_main_window(monkeypatch)
    reloaded_window._load_config_from_file(str(config_path), preset_mode=True)

    assert reloaded_window._postprocess_panel.g_refinekit.isHidden() is False
    assert (
        reloaded_window._postprocess_panel.chk_prompt_open_refinekit.isChecked() is True
    )
    reloaded_window.close()


def test_video_autoload_restores_pose_keypoint_groups_and_headtail_model(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"")
    skeleton_path = tmp_path / "skeleton.json"
    skeleton_path.write_text(
        json.dumps(
            {
                "keypoint_names": ["head", "thorax", "abdomen", "tail"],
                "skeleton_edges": [[0, 1], [1, 2], [2, 3]],
            }
        ),
        encoding="utf-8",
    )

    window = _make_main_window(monkeypatch)
    window.current_video_path = str(video_path)
    window._setup_panel.file_line.setText(str(video_path))
    window._identity_panel.g_headtail.setChecked(True)
    window._identity_panel._refresh_yolo_headtail_model_combo()
    selected_model = _select_first_model_with_suffix(
        window._identity_panel.combo_yolo_headtail_model,
        ".pth",
    )
    assert selected_model
    window._identity_panel.chk_enable_pose_extractor.setChecked(True)
    window._identity_panel.line_pose_skeleton_file.setText(str(skeleton_path))
    window._identity_panel._refresh_pose_direction_keypoint_lists()
    window._set_pose_group_selection(
        window._identity_panel.list_pose_direction_anterior,
        ["head", "thorax"],
    )
    window._set_pose_group_selection(
        window._identity_panel.list_pose_direction_posterior,
        ["tail"],
    )
    window._apply_pose_keypoint_selection_constraints("anterior")

    config_path = tmp_path / "sample_config.json"
    assert window.save_config(preset_mode=False, preset_path=str(config_path))
    saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_cfg["pose_direction_anterior_keypoints"] == ["head", "thorax"]
    assert saved_cfg["pose_direction_posterior_keypoints"] == ["tail"]
    window.close()

    reloaded_window = _make_main_window(monkeypatch)

    def _fake_init_video_player(_path: str) -> None:
        reloaded_window.video_total_frames = 120
        reloaded_window._setup_panel.spin_start_frame.setMaximum(119)
        reloaded_window._setup_panel.spin_end_frame.setMaximum(119)
        reloaded_window._setup_panel.spin_start_frame.setEnabled(True)
        reloaded_window._setup_panel.spin_end_frame.setEnabled(True)

    monkeypatch.setattr(reloaded_window, "_init_video_player", _fake_init_video_player)

    reloaded_window._setup_video_file(str(video_path))

    assert (
        reloaded_window._identity_panel._get_configured_yolo_headtail_model_path()
        == selected_model
    )
    assert reloaded_window._parse_pose_direction_anterior_keypoints() == [
        "head",
        "thorax",
    ]
    assert reloaded_window._parse_pose_direction_posterior_keypoints() == ["tail"]
    reloaded_window.close()


def test_remove_buttons_delete_only_the_selected_tracker_models(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    models_root = _seed_trackerkit_model_repository(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "hydra_suite.trackerkit.gui.orchestrators.config.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Yes,
    )
    monkeypatch.setattr(
        "hydra_suite.trackerkit.gui.orchestrators.config.QMessageBox.warning",
        lambda *_args, **_kwargs: None,
    )

    window = _make_main_window(monkeypatch)
    window._set_yolo_model_selection("obb/direct_remove.pt")
    window._detection_panel.btn_remove_yolo_model.click()

    registry = _load_seeded_model_registry(models_root)
    assert not (models_root / "obb/direct_remove.pt").exists()
    assert "obb/direct_remove.pt" not in registry
    assert (models_root / "obb/direct_keep.pt").exists()
    assert "obb/direct_keep.pt" in registry

    window._set_yolo_detect_model_selection("detection/seq_detect_remove.pt")
    window._detection_panel.btn_remove_yolo_detect_model.click()
    registry = _load_seeded_model_registry(models_root)
    assert not (models_root / "detection/seq_detect_remove.pt").exists()
    assert "detection/seq_detect_remove.pt" not in registry
    assert (models_root / "detection/seq_detect_keep.pt").exists()

    window._set_yolo_crop_obb_model_selection("obb/cropped/seq_crop_remove.pt")
    window._detection_panel.btn_remove_yolo_crop_obb_model.click()
    registry = _load_seeded_model_registry(models_root)
    assert not (models_root / "obb/cropped/seq_crop_remove.pt").exists()
    assert "obb/cropped/seq_crop_remove.pt" not in registry
    assert (models_root / "obb/cropped/seq_crop_keep.pt").exists()

    window._identity_panel.g_headtail.setChecked(True)
    window._set_yolo_headtail_model_selection(
        "classification/orientation/YOLO/headtail_remove.pt"
    )
    window._identity_panel.btn_remove_yolo_headtail_model.click()
    registry = _load_seeded_model_registry(models_root)
    assert not (
        models_root / "classification/orientation/YOLO/headtail_remove.pt"
    ).exists()
    assert "classification/orientation/YOLO/headtail_remove.pt" not in registry
    assert (models_root / "classification/orientation/YOLO/headtail_keep.pt").exists()
    window.close()


def test_remove_buttons_delete_selected_pose_and_sleap_models_only(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    models_root = _seed_trackerkit_model_repository(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "hydra_suite.trackerkit.gui.orchestrators.config.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Yes,
    )
    monkeypatch.setattr(
        "hydra_suite.trackerkit.gui.orchestrators.config.QMessageBox.warning",
        lambda *_args, **_kwargs: None,
    )

    window = _make_main_window(monkeypatch)
    window._identity_panel.g_pose_runtime.setChecked(True)
    window._identity_panel.combo_pose_model_type.setCurrentText("YOLO")
    window._set_pose_model_path_for_backend(
        "pose/YOLO/pose_remove.pt",
        backend="yolo",
        update_combo=True,
    )
    window._identity_panel.btn_remove_pose_model.click()
    assert not (models_root / "pose/YOLO/pose_remove.pt").exists()
    assert (models_root / "pose/YOLO/pose_keep.pt").exists()

    window._identity_panel.combo_pose_model_type.setCurrentText("SLEAP")
    window._set_pose_model_path_for_backend(
        "pose/SLEAP/sleap_remove",
        backend="sleap",
        update_combo=True,
    )
    window._identity_panel.btn_remove_pose_model.click()
    assert not (models_root / "pose/SLEAP/sleap_remove").exists()
    assert (models_root / "pose/SLEAP/sleap_keep").exists()
    window.close()


def test_remove_button_deletes_only_selected_cnn_identity_model(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    models_root = _seed_trackerkit_model_repository(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "hydra_suite.trackerkit.gui.orchestrators.config.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Yes,
    )
    monkeypatch.setattr(
        "hydra_suite.trackerkit.gui.orchestrators.config.QMessageBox.warning",
        lambda *_args, **_kwargs: None,
    )

    window = _make_main_window(monkeypatch)
    window._detection_panel.combo_detection_method.setCurrentIndex(1)
    window._sync_individual_analysis_mode_ui()
    window._identity_panel.g_identity.setChecked(True)
    row = window._identity_panel._add_cnn_classifier_row()
    idx = row.combo_model.findData("classification/identity/cnn_remove.pt")
    assert idx >= 0
    row.combo_model.setCurrentIndex(idx)
    assert row.btn_remove_model.isEnabled() is True
    row.btn_remove_model.click()

    registry = _load_seeded_model_registry(models_root)
    assert not (models_root / "classification/identity/cnn_remove.pt").exists()
    assert "classification/identity/cnn_remove.pt" not in registry
    assert (models_root / "classification/identity/cnn_keep.pt").exists()
    assert row.combo_model.findData("classification/identity/cnn_remove.pt") == -1
    assert row.combo_model.findData("classification/identity/cnn_keep.pt") >= 0
    window.close()


def test_preview_detection_restores_analyze_individual_controls(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    _seed_trackerkit_model_repository(tmp_path, monkeypatch)

    window = _make_main_window(monkeypatch)
    window.preview_frame_original = np.zeros((24, 24, 3), dtype=np.uint8)
    window._detection_panel.combo_detection_method.setCurrentIndex(1)
    window._sync_individual_analysis_mode_ui()

    window._identity_panel.g_identity.setChecked(True)
    window._identity_panel.g_apriltags.setChecked(True)
    window._identity_panel.g_headtail.setChecked(True)
    window._identity_panel.g_pose_runtime.setChecked(True)
    window._set_yolo_headtail_model_selection(
        "classification/orientation/YOLO/headtail_keep.pt"
    )
    window._set_pose_model_path_for_backend(
        "pose/YOLO/pose_keep.pt",
        backend="yolo",
        update_combo=True,
    )

    assert window._identity_panel.spin_identity_match_bonus.isEnabled() is True
    assert window._identity_panel.spin_identity_mismatch_penalty.isEnabled() is True
    assert window._identity_panel.combo_apriltag_family.isEnabled() is True
    assert window._identity_panel.spin_apriltag_decimate.isEnabled() is True
    assert window._identity_panel.combo_pose_model_type.isEnabled() is True
    assert window._identity_panel.combo_pose_model.isEnabled() is True
    assert window._identity_panel.spin_pose_min_kpt_conf_valid.isEnabled() is True
    assert window._identity_panel.spin_pose_batch.isEnabled() is True
    assert window._identity_panel.btn_remove_pose_model.isEnabled() is True
    assert window._identity_panel.btn_remove_yolo_headtail_model.isEnabled() is True

    window._session_orch._set_preview_test_running(True)

    assert window._identity_panel.spin_identity_match_bonus.isEnabled() is False
    assert window._identity_panel.btn_remove_pose_model.isEnabled() is False

    window._session_orch._set_preview_test_running(False)

    assert window._identity_panel.spin_identity_match_bonus.isEnabled() is True
    assert window._identity_panel.spin_identity_mismatch_penalty.isEnabled() is True
    assert window._identity_panel.combo_apriltag_family.isEnabled() is True
    assert window._identity_panel.spin_apriltag_decimate.isEnabled() is True
    assert window._identity_panel.combo_pose_model_type.isEnabled() is True
    assert window._identity_panel.combo_pose_model.isEnabled() is True
    assert window._identity_panel.spin_pose_min_kpt_conf_valid.isEnabled() is True
    assert window._identity_panel.spin_pose_batch.isEnabled() is True
    assert window._identity_panel.btn_remove_pose_model.isEnabled() is True
    assert window._identity_panel.btn_remove_yolo_headtail_model.isEnabled() is True
    assert window.btn_test_detection.isEnabled() is True
    window.close()


def test_tier_combo_exists_and_old_per_stage_combos_removed(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
) -> None:
    """combo_runtime_tier replaces the three old per-stage runtime combos."""
    from PySide6.QtWidgets import QComboBox, QLabel

    window = _make_main_window(monkeypatch)
    panel = window._setup_panel
    assert hasattr(panel, "combo_runtime_tier"), "combo_runtime_tier missing"
    assert isinstance(panel.combo_runtime_tier, QComboBox)
    assert panel.combo_runtime_tier.count() >= 1
    assert hasattr(panel, "lbl_runtime_fallback"), "lbl_runtime_fallback missing"
    assert isinstance(panel.lbl_runtime_fallback, QLabel)
    assert not hasattr(
        panel, "combo_compute_runtime"
    ), "old combo_compute_runtime present"
    assert not hasattr(
        panel, "combo_headtail_runtime"
    ), "old combo_headtail_runtime present"
    assert not hasattr(panel, "combo_cnn_runtime"), "old combo_cnn_runtime present"
    window.close()


def test_setup_panel_has_no_pose_runtime_control(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    """Pose runtime has no UI control at all — it's fully derived from Compute tier."""
    _seed_trackerkit_model_repository(tmp_path, monkeypatch)
    window = _make_main_window(monkeypatch)
    assert not hasattr(window._setup_panel, "lbl_pose_runtime_flavor")
    assert not hasattr(window._setup_panel, "combo_pose_runtime_flavor")
    assert not hasattr(window._identity_panel, "combo_pose_runtime_flavor")
    window.close()


def test_setup_panel_single_tier_combo_always_visible(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    """The single compute-tier combo is always visible regardless of identity config."""
    _seed_trackerkit_model_repository(tmp_path, monkeypatch)
    window = _make_main_window(monkeypatch)
    panel = window._setup_panel

    window._identity_panel.g_identity.setChecked(False)
    window._sync_individual_analysis_mode_ui()
    assert panel.combo_runtime_tier.isHidden() is False

    window._identity_panel.g_identity.setChecked(True)
    window._identity_panel.g_headtail.setChecked(True)
    _select_first_nonempty_model(window._identity_panel.combo_yolo_headtail_model)
    window._sync_individual_analysis_mode_ui()
    assert panel.combo_runtime_tier.isHidden() is False
    window.close()


def test_saved_config_omits_retired_pose_runtime_flavor(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    """Runtime Gen-2 (FT2): pose runtime is not stored; it is fully tier-derived.

    The legacy ``pose_runtime_flavor`` / ``pose_sleap_device`` stored-config fields
    are retired -- only ``runtime_tier`` is persisted, and the pose device is
    resolved from it downstream.
    """
    window = _make_main_window(monkeypatch)

    window._identity_panel.chk_enable_pose_extractor.setChecked(True)
    window._identity_panel.combo_pose_model_type.setCurrentText("SLEAP")

    idx = window._setup_panel.combo_runtime_tier.findData("cpu")
    assert idx >= 0
    window._setup_panel.combo_runtime_tier.setCurrentIndex(idx)

    config_path = tmp_path / "pose_runtime.json"
    assert window.save_config(preset_mode=True, preset_path=str(config_path))
    saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert "pose_runtime_flavor" not in saved_cfg
    assert "pose_sleap_device" not in saved_cfg
    assert saved_cfg["runtime_tier"] == "cpu"
    window.close()


def test_saved_config_preserves_runtime_tier(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    """Saving config persists the selected runtime_tier."""
    window = _make_main_window(monkeypatch)
    combo = window._setup_panel.combo_runtime_tier
    # Select the first tier (cpu) explicitly.
    combo.setCurrentIndex(0)
    first_tier = combo.currentData()

    config_path = tmp_path / "tier_runtime.json"
    assert window.save_config(preset_mode=True, preset_path=str(config_path))
    saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_cfg["runtime_tier"] == first_tier
    window.close()


def test_headtail_runtime_derives_from_tier_combo(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    """The retired per-stage runtime combos are gone; the OBB backend is
    resolved from the selected tier (Runtime Gen-2)."""
    from hydra_suite.runtime.resolver import ResolvedBackend

    _seed_trackerkit_model_repository(tmp_path, monkeypatch)
    window = _make_main_window(monkeypatch)

    # headtail and cnn combos no longer exist on setup panel.
    assert not hasattr(window._setup_panel, "combo_headtail_runtime")
    assert not hasattr(window._setup_panel, "combo_cnn_runtime")
    # There is a single tier-derived resolved OBB backend for all stages.
    assert isinstance(window._resolved_obb_backend(), ResolvedBackend)
    window.close()


def test_pose_video_overlay_customization_controls_remain_visible(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    _seed_trackerkit_model_repository(tmp_path, monkeypatch)
    window = _make_main_window(monkeypatch)
    window._detection_panel.combo_detection_method.setCurrentIndex(1)
    window._identity_panel.chk_enable_pose_extractor.setChecked(True)
    window._identity_panel.combo_pose_model_type.setCurrentText("YOLO")
    _select_first_nonempty_model(window._identity_panel.combo_pose_model)
    window._postprocess_panel.check_video_output.setChecked(True)
    window._postprocess_panel.check_video_show_pose.setChecked(True)

    window._sync_video_pose_overlay_controls()

    assert window._postprocess_panel.combo_video_pose_color_mode.isHidden() is False
    assert window._postprocess_panel.spin_video_pose_point_radius.isHidden() is False
    assert window._postprocess_panel.spin_video_pose_point_thickness.isHidden() is False
    assert window._postprocess_panel.spin_video_pose_line_thickness.isHidden() is False
    assert window._postprocess_panel.lbl_video_pose_disabled_hint.isHidden() is False
    window.close()


def test_confidence_density_toggle_roundtrip_updates_visibility(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    window = _make_main_window(monkeypatch)

    assert not window._tracking_panel.g_density.isHidden()
    assert (
        window._tracking_panel.chk_export_confidence_density_video.isChecked() is False
    )

    window._tracking_panel.chk_enable_confidence_density_map.setChecked(False)
    window._tracking_panel.chk_export_confidence_density_video.setChecked(False)

    assert window._tracking_panel.g_density.isHidden()
    assert window.get_parameters_dict()["ENABLE_CONFIDENCE_DENSITY_MAP"] is False
    assert window.get_parameters_dict()["EXPORT_CONFIDENCE_DENSITY_VIDEO"] is False

    config_path = tmp_path / "confidence_density_toggle.json"
    assert window.save_config(preset_mode=True, preset_path=str(config_path))
    saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_cfg["enable_confidence_density_map"] is False
    assert saved_cfg["export_confidence_density_video"] is False
    window.close()

    reloaded_window = _make_main_window(monkeypatch)
    reloaded_window._load_config_from_file(str(config_path), preset_mode=True)

    assert (
        reloaded_window._tracking_panel.chk_enable_confidence_density_map.isChecked()
        is False
    )
    assert (
        reloaded_window._tracking_panel.chk_export_confidence_density_video.isChecked()
        is False
    )
    assert reloaded_window._tracking_panel.g_density.isHidden()

    reloaded_window._tracking_panel.chk_enable_confidence_density_map.setChecked(True)
    reloaded_window._tracking_panel.chk_export_confidence_density_video.setChecked(True)

    assert not reloaded_window._tracking_panel.g_density.isHidden()
    assert (
        reloaded_window.get_parameters_dict()["EXPORT_CONFIDENCE_DENSITY_VIDEO"] is True
    )
    reloaded_window.close()


def test_final_media_video_paths_are_split_from_individual_crop_paths(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"")

    window = _make_main_window(monkeypatch)
    window.current_video_path = str(video_path)
    window._setup_panel.file_line.setText(str(video_path))
    window._detection_panel.combo_detection_method.setCurrentIndex(1)
    window._dataset_panel.chk_enable_individual_dataset.setChecked(False)
    window._sync_individual_analysis_mode_ui()

    params = window.get_parameters_dict()

    assert window._dataset_panel.g_individual_dataset.isHidden() is False
    assert window._dataset_panel.g_oriented_videos.isHidden() is False
    assert window._dataset_panel.ind_output_group.isHidden() is True
    assert params["INDIVIDUAL_DATASET_OUTPUT_DIR"] == str(
        tmp_path / "sample_datasets" / "individual_crops"
    )
    assert params["FINAL_MEDIA_EXPORT_VIDEO_OUTPUT_DIR"] == str(
        tmp_path / "sample_datasets" / "oriented_videos"
    )
    window.close()


def test_final_media_video_postprocess_controls_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    window = _make_main_window(monkeypatch)
    window._detection_panel.combo_detection_method.setCurrentIndex(1)
    window._sync_individual_analysis_mode_ui()

    window._dataset_panel.chk_enable_individual_dataset.setChecked(True)
    window._dataset_panel.chk_suppress_foreign_obb_individual_dataset.setChecked(True)
    window._dataset_panel.chk_generate_individual_track_videos.setChecked(True)
    window._dataset_panel.chk_suppress_foreign_obb_oriented_videos.setChecked(False)
    window._dataset_panel.chk_fix_oriented_video_direction_flips.setChecked(True)
    window._dataset_panel.spin_oriented_video_heading_flip_burst.setValue(7)
    window._dataset_panel.chk_enable_oriented_video_affine_stabilization.setChecked(
        True
    )
    window._dataset_panel.spin_oriented_video_stabilization_window.setValue(9)

    params = window.get_parameters_dict()
    assert params["SUPPRESS_FOREIGN_OBB_DATASET"] is True
    assert params["SUPPRESS_FOREIGN_OBB_ORIENTED_VIDEO"] is False
    assert params["FINAL_MEDIA_EXPORT_FIX_DIRECTION_FLIPS"] is True
    assert params["FINAL_MEDIA_EXPORT_HEADING_FLIP_MAX_BURST"] == 7
    assert params["FINAL_MEDIA_EXPORT_ENABLE_AFFINE_STABILIZATION"] is True
    assert params["FINAL_MEDIA_EXPORT_STABILIZATION_WINDOW"] == 9

    config_path = tmp_path / "oriented_video_postprocess_roundtrip.json"
    assert window.save_config(preset_mode=True, preset_path=str(config_path))
    saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_cfg["suppress_foreign_obb_individual_dataset"] is True
    assert saved_cfg["suppress_foreign_obb_oriented_videos"] is False
    assert saved_cfg["final_media_export_fix_direction_flips"] is True
    assert saved_cfg["final_media_export_heading_flip_burst"] == 7
    assert saved_cfg["final_media_export_enable_affine_stabilization"] is True
    assert saved_cfg["final_media_export_stabilization_window"] == 9
    window.close()

    reloaded_window = _make_main_window(monkeypatch)
    reloaded_window._load_config_from_file(str(config_path), preset_mode=True)

    assert (
        reloaded_window._dataset_panel.chk_fix_oriented_video_direction_flips.isChecked()
        is True
    )
    assert (
        reloaded_window._dataset_panel.chk_suppress_foreign_obb_individual_dataset.isChecked()
        is True
    )
    assert (
        reloaded_window._dataset_panel.chk_suppress_foreign_obb_oriented_videos.isChecked()
        is False
    )
    assert (
        reloaded_window._dataset_panel.spin_oriented_video_heading_flip_burst.value()
        == 7
    )
    assert (
        reloaded_window._dataset_panel.chk_enable_oriented_video_affine_stabilization.isChecked()
        is True
    )
    assert (
        reloaded_window._dataset_panel.spin_oriented_video_stabilization_window.value()
        == 9
    )
    reloaded_window.close()


def test_realtime_workflow_and_final_image_export_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    window = _make_main_window(monkeypatch)
    window._detection_panel.combo_detection_method.setCurrentIndex(1)
    window._sync_individual_analysis_mode_ui()

    window._setup_panel.chk_use_cached_detections.setChecked(True)
    window._setup_panel.chk_realtime_mode.setChecked(True)
    window._dataset_panel.chk_enable_individual_dataset.setChecked(True)

    assert window._setup_panel.chk_use_cached_detections.isChecked() is False
    assert window._setup_panel.chk_use_cached_detections.isEnabled() is False

    params = window.get_parameters_dict()
    assert params["TRACKING_REALTIME_MODE"] is True
    assert params["TRACKING_WORKFLOW_MODE"] == "realtime"
    assert params["ENABLE_INDIVIDUAL_IMAGE_SAVE"] is False
    assert params["ENABLE_INDIVIDUAL_DATASET"] is False
    assert params["EXPORT_FINAL_CANONICAL_IMAGES"] is True

    config_path = tmp_path / "realtime_workflow_roundtrip.json"
    assert window.save_config(preset_mode=True, preset_path=str(config_path))
    saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_cfg["realtime_tracking_mode"] is True
    assert saved_cfg["tracking_workflow_mode"] == "realtime"
    assert saved_cfg["export_final_canonical_images"] is True
    window.close()

    reloaded_window = _make_main_window(monkeypatch)
    reloaded_window._load_config_from_file(str(config_path), preset_mode=True)

    assert reloaded_window._setup_panel.chk_realtime_mode.isChecked() is True
    assert reloaded_window._setup_panel.chk_use_cached_detections.isChecked() is False
    assert reloaded_window._setup_panel.chk_use_cached_detections.isEnabled() is False
    assert (
        reloaded_window._dataset_panel.chk_enable_individual_dataset.isChecked() is True
    )
    reloaded_window.close()


def test_realtime_batch_policy_clamps_identity_controls_to_animal_count(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    _seed_trackerkit_model_repository(tmp_path, monkeypatch)

    window = _make_main_window(monkeypatch)
    window._detection_panel.combo_detection_method.setCurrentIndex(1)
    window._sync_individual_analysis_mode_ui()

    row = window._identity_panel._add_cnn_classifier_row()
    row.spin_batch.setValue(32)
    window._identity_panel.spin_pose_batch.setValue(16)
    window._setup_panel.spin_max_targets.setValue(3)
    window._setup_panel.chk_realtime_mode.setChecked(True)

    assert window._detection_panel.lbl_batch_policy_notice.isHidden() is False
    assert (
        "one frame at a time" in window._detection_panel.lbl_batch_policy_notice.text()
    )
    assert window._detection_panel.spin_detection_batch_size.isEnabled() is False
    assert window._identity_panel.spin_pose_batch.maximum() == 3
    assert window._identity_panel.spin_pose_batch.value() == 3
    assert row.spin_batch.maximum() == 3
    assert row.spin_batch.value() == 3
    assert window._identity_panel.lbl_individual_batch_notice.isHidden() is False
    assert (
        "capped to 3 animal(s) per frame"
        in window._identity_panel.lbl_individual_batch_notice.text()
    )
    window.close()


def test_realtime_sequential_mode_keeps_crop_batch_setting_visible(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    _seed_trackerkit_model_repository(tmp_path, monkeypatch)

    window = _make_main_window(monkeypatch)
    window._detection_panel.combo_detection_method.setCurrentIndex(1)
    window._detection_panel.combo_yolo_obb_mode.setCurrentIndex(1)
    window._detection_panel.spin_yolo_seq_individual_batch_size.setValue(7)
    window._setup_panel.chk_realtime_mode.setChecked(True)

    assert window._detection_panel.spin_detection_batch_size.isEnabled() is False
    assert (
        window._detection_panel.spin_yolo_seq_individual_batch_size.isEnabled() is True
    )
    assert (
        "Sequential stage-2 crop batching still uses the Stage-2 crop batch setting"
        in window._detection_panel.lbl_batch_policy_notice.text()
    )
    window.close()


def test_realtime_direct_mode_exposes_micro_batch_controls(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    _seed_trackerkit_model_repository(tmp_path, monkeypatch)

    window = _make_main_window(monkeypatch)
    window._detection_panel.combo_detection_method.setCurrentIndex(1)
    window._setup_panel.chk_realtime_mode.setChecked(True)

    assert (
        window._detection_panel.chk_enable_realtime_yolo_micro_batching.isEnabled()
        is True
    )
    assert (
        window._detection_panel.spin_realtime_yolo_micro_batch_size.isEnabled() is False
    )

    window._detection_panel.chk_enable_realtime_yolo_micro_batching.setChecked(True)
    window._detection_panel.spin_realtime_yolo_micro_batch_size.setValue(4)
    window._detection_panel._sync_live_detection_batch_controls()

    assert (
        window._detection_panel.spin_realtime_yolo_micro_batch_size.isEnabled() is True
    )
    assert (
        "queues up to 4 frame(s)"
        in window._detection_panel.lbl_batch_policy_notice.text()
    )
    window.close()


def test_realtime_micro_batch_roundtrip_persists(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    _seed_trackerkit_model_repository(tmp_path, monkeypatch)

    window = _make_main_window(monkeypatch)
    window._detection_panel.combo_detection_method.setCurrentIndex(1)
    window._detection_panel.chk_enable_realtime_yolo_micro_batching.setChecked(True)
    window._detection_panel.spin_realtime_yolo_micro_batch_size.setValue(3)

    config_path = tmp_path / "realtime_micro_batch_roundtrip.json"
    assert window.save_config(preset_mode=True, preset_path=str(config_path))
    saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_cfg["enable_realtime_yolo_micro_batching"] is True
    assert saved_cfg["realtime_yolo_micro_batch_size"] == 3
    window.close()

    reloaded_window = _make_main_window(monkeypatch)
    reloaded_window._load_config_from_file(str(config_path), preset_mode=True)

    assert (
        reloaded_window._detection_panel.chk_enable_realtime_yolo_micro_batching.isChecked()
        is True
    )
    assert (
        reloaded_window._detection_panel.spin_realtime_yolo_micro_batch_size.value()
        == 3
    )
    reloaded_window.close()


def test_sequential_crop_batch_roundtrip_persists(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    _seed_trackerkit_model_repository(tmp_path, monkeypatch)

    window = _make_main_window(monkeypatch)
    window._detection_panel.combo_detection_method.setCurrentIndex(1)
    window._detection_panel.combo_yolo_obb_mode.setCurrentIndex(1)
    window._detection_panel.spin_yolo_seq_individual_batch_size.setValue(9)

    config_path = tmp_path / "sequential_crop_batch_roundtrip.json"
    assert window.save_config(preset_mode=True, preset_path=str(config_path))
    saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_cfg["yolo_seq_individual_batch_size"] == 9
    window.close()

    reloaded_window = _make_main_window(monkeypatch)
    reloaded_window._load_config_from_file(str(config_path), preset_mode=True)

    assert (
        reloaded_window._detection_panel.spin_yolo_seq_individual_batch_size.value()
        == 9
    )
    reloaded_window.close()


def test_legacy_shared_suppress_setting_populates_both_export_toggles(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "legacy_shared_suppress.json"
    config_path.write_text(
        json.dumps(
            {
                "suppress_foreign_obb_dataset": True,
                "enable_individual_image_save": False,
                "generate_oriented_track_videos": False,
            }
        ),
        encoding="utf-8",
    )

    window = _make_main_window(monkeypatch)
    window._load_config_from_file(str(config_path), preset_mode=True)

    assert (
        window._dataset_panel.chk_suppress_foreign_obb_individual_dataset.isChecked()
        is True
    )
    assert (
        window._dataset_panel.chk_suppress_foreign_obb_oriented_videos.isChecked()
        is True
    )
    window.close()


def test_session_summary_refinekit_prompt_respects_toggle_and_batch_mode(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
) -> None:
    window = _make_main_window(monkeypatch)
    window.current_video_path = "/tmp/video.mp4"

    prompt_calls: list[str] = []
    monkeypatch.setattr(
        "hydra_suite.trackerkit.gui.orchestrators.tracking.QMessageBox.information",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "hydra_suite.trackerkit.gui.orchestrators.tracking.QMessageBox.question",
        lambda *_args, **_kwargs: prompt_calls.append("asked") or QMessageBox.No,
    )

    window._postprocess_panel.chk_prompt_open_refinekit.setChecked(False)
    window._tracking_orch._show_session_summary()
    assert prompt_calls == []

    window._postprocess_panel.chk_prompt_open_refinekit.setChecked(True)
    window._tracking_orch._show_session_summary()
    assert prompt_calls == ["asked"]

    window._setup_panel.g_batch.setChecked(True)
    window._tracking_orch._show_session_summary()
    assert prompt_calls == ["asked"]
    window.close()


def test_get_parameters_dict_commits_pending_frame_range_edit(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
) -> None:
    window = _make_main_window(monkeypatch)
    window._setup_panel.spin_start_frame.setEnabled(True)
    window._setup_panel.spin_end_frame.setEnabled(True)
    window._setup_panel.spin_start_frame.setMaximum(500)
    window._setup_panel.spin_end_frame.setMaximum(500)
    window._setup_panel.spin_start_frame.setValue(40)
    window._setup_panel.spin_end_frame.setValue(80)

    window._setup_panel.spin_end_frame.lineEdit().setText("25")
    params = window.get_parameters_dict()

    assert params["START_FRAME"] == 25
    assert params["END_FRAME"] == 25
    window.close()


def test_advanced_config_defaults_include_identity_decoder_tuning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    advanced_path = tmp_path / "advanced_config.json"

    monkeypatch.setattr(
        "hydra_suite.paths.get_advanced_config_path",
        lambda: advanced_path,
    )

    advanced = ConfigOrchestrator._load_advanced_config(object())

    assert advanced["identity_offline_split_trajectories"] is False
    assert advanced["identity_offline_split_min_conf"] == pytest.approx(0.75)
    assert advanced["identity_offline_split_min_margin"] == pytest.approx(0.2)
    assert advanced["identity_offline_split_min_frames"] == 3
    assert advanced["identity_offline_split_max_bridge_frames"] == 6
    assert advanced["identity_offline_ilp_time_limit"] == pytest.approx(30.0)
    assert advanced["identity_offline_ilp_rel_gap"] == pytest.approx(1e-6)
    assert advanced["identity_respawn_prior_strength"] == pytest.approx(0.75)
    assert advanced["identity_respawn_prior_decay"] == pytest.approx(0.97)
    assert advanced["identity_respawn_prior_max_gap"] == 120

    saved = json.loads(advanced_path.read_text(encoding="utf-8"))
    assert saved["identity_offline_split_trajectories"] is False
    assert saved["identity_offline_split_min_conf"] == pytest.approx(0.75)
    assert saved["identity_respawn_prior_max_gap"] == 120


def test_get_parameters_dict_exposes_identity_decoder_advanced_overrides(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
) -> None:
    window = _make_main_window(
        monkeypatch,
        advanced_config={
            "identity_offline_split_trajectories": True,
            "identity_offline_split_min_conf": 0.81,
            "identity_offline_split_min_margin": 0.27,
            "identity_offline_split_min_frames": 5,
            "identity_offline_split_max_bridge_frames": 8,
            "identity_offline_ilp_time_limit": 12.5,
            "identity_offline_ilp_rel_gap": 1e-4,
            "identity_respawn_prior_strength": 0.62,
            "identity_respawn_prior_decay": 0.93,
            "identity_respawn_prior_max_gap": 44,
        },
    )

    params = window.get_parameters_dict()

    assert params["IDENTITY_OFFLINE_SPLIT_TRAJECTORIES"] is True
    assert params["IDENTITY_OFFLINE_SPLIT_MIN_CONF"] == pytest.approx(0.81)
    assert params["IDENTITY_OFFLINE_SPLIT_MIN_MARGIN"] == pytest.approx(0.27)
    assert params["IDENTITY_OFFLINE_SPLIT_MIN_FRAMES"] == 5
    assert params["IDENTITY_OFFLINE_SPLIT_MAX_BRIDGE_FRAMES"] == 8
    assert params["IDENTITY_OFFLINE_ILP_TIME_LIMIT"] == pytest.approx(12.5)
    assert params["IDENTITY_OFFLINE_ILP_REL_GAP"] == pytest.approx(1e-4)
    assert params["IDENTITY_RESPAWN_PRIOR_STRENGTH"] == pytest.approx(0.62)
    assert params["IDENTITY_RESPAWN_PRIOR_DECAY"] == pytest.approx(0.93)
    assert params["IDENTITY_RESPAWN_PRIOR_MAX_GAP"] == 44
    assert params["ADVANCED_CONFIG"]["identity_respawn_prior_max_gap"] == 44
    window.close()


def test_identity_decoder_tuning_controls_roundtrip_through_tracker_config(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    window = _make_main_window(
        monkeypatch,
        advanced_config={
            "identity_offline_split_trajectories": True,
            "identity_offline_split_min_conf": 0.82,
            "identity_respawn_prior_max_gap": 77,
        },
    )

    assert window._identity_panel.chk_identity_offline_split_trajectories.isChecked()
    assert (
        window._identity_panel.spin_identity_offline_split_min_conf.value()
        == pytest.approx(0.82)
    )
    assert window._identity_panel.spin_identity_respawn_prior_max_gap.value() == 77

    window._identity_panel.chk_identity_offline_split_trajectories.setChecked(False)
    window._identity_panel.spin_identity_offline_split_min_conf.setValue(0.79)
    window._identity_panel.spin_identity_offline_split_min_margin.setValue(0.24)
    window._identity_panel.spin_identity_offline_split_min_frames.setValue(4)
    window._identity_panel.spin_identity_offline_split_max_bridge_frames.setValue(9)
    window._identity_panel.spin_identity_offline_ilp_time_limit.setValue(15.0)
    window._identity_panel.spin_identity_offline_ilp_rel_gap.setValue(0.0025)
    window._identity_panel.spin_identity_respawn_prior_strength.setValue(0.66)
    window._identity_panel.spin_identity_respawn_prior_decay.setValue(0.91)
    window._identity_panel.spin_identity_respawn_prior_max_gap.setValue(42)

    config_path = tmp_path / "identity_decoder_tuning_roundtrip.json"
    assert window.save_config(preset_mode=True, preset_path=str(config_path))
    saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))

    assert saved_cfg["identity_offline_split_trajectories"] is False
    assert saved_cfg["identity_offline_split_min_conf"] == pytest.approx(0.79)
    assert saved_cfg["identity_offline_split_min_margin"] == pytest.approx(0.24)
    assert saved_cfg["identity_offline_split_min_frames"] == 4
    assert saved_cfg["identity_offline_split_max_bridge_frames"] == 9
    assert saved_cfg["identity_offline_ilp_time_limit"] == pytest.approx(15.0)
    assert saved_cfg["identity_offline_ilp_rel_gap"] == pytest.approx(0.0025)
    assert saved_cfg["identity_respawn_prior_strength"] == pytest.approx(0.66)
    assert saved_cfg["identity_respawn_prior_decay"] == pytest.approx(0.91)
    assert saved_cfg["identity_respawn_prior_max_gap"] == 42
    window.close()

    reloaded_window = _make_main_window(monkeypatch)
    reloaded_window._load_config_from_file(str(config_path), preset_mode=True)

    assert (
        not reloaded_window._identity_panel.chk_identity_offline_split_trajectories.isChecked()
    )
    assert (
        reloaded_window._identity_panel.spin_identity_offline_split_min_conf.value()
        == pytest.approx(0.79)
    )
    assert (
        reloaded_window._identity_panel.spin_identity_offline_split_min_margin.value()
        == pytest.approx(0.24)
    )
    assert (
        reloaded_window._identity_panel.spin_identity_offline_split_min_frames.value()
        == 4
    )
    assert (
        reloaded_window._identity_panel.spin_identity_offline_split_max_bridge_frames.value()
        == 9
    )
    assert (
        reloaded_window._identity_panel.spin_identity_offline_ilp_time_limit.value()
        == pytest.approx(15.0)
    )
    assert (
        reloaded_window._identity_panel.spin_identity_offline_ilp_rel_gap.value()
        == pytest.approx(0.0025)
    )
    assert (
        reloaded_window._identity_panel.spin_identity_respawn_prior_strength.value()
        == pytest.approx(0.66)
    )
    assert (
        reloaded_window._identity_panel.spin_identity_respawn_prior_decay.value()
        == pytest.approx(0.91)
    )
    assert (
        reloaded_window._identity_panel.spin_identity_respawn_prior_max_gap.value()
        == 42
    )
    reloaded_window.close()


def test_trail_history_special_values_update_overlay_toggle_and_clamp(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
) -> None:
    window = _make_main_window(monkeypatch)
    window.video_total_frames = 240
    window._session_orch._sync_trail_history_bounds()

    window._setup_panel.chk_show_trajectories.setChecked(True)
    window._setup_panel.spin_traj_hist.setValue(0)
    assert window._setup_panel.chk_show_trajectories.isChecked() is False

    window._setup_panel.spin_traj_hist.setValue(-1)
    assert window._setup_panel.chk_show_trajectories.isChecked() is True

    window._setup_panel.spin_traj_hist.setValue(999)
    params = window.get_parameters_dict()

    assert params["TRAJECTORY_HISTORY_SECONDS"] == 240
    assert params["SHOW_TRAJECTORIES"] is True
    window.close()


def test_bg_parameter_helper_applies_extended_detection_params(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"")

    class _FakeDialog:
        def __init__(self, _video_path: str, _params: dict[str, object], _parent=None):
            pass

        def exec(self) -> int:
            return QDialog.Accepted

        def get_selected_params(self) -> dict[str, object]:
            return {
                "BRIGHTNESS": 12,
                "CONTRAST": 1.35,
                "GAMMA": 0.8,
                "DARK_ON_LIGHT_BACKGROUND": False,
                "BACKGROUND_PRIME_FRAMES": 60,
                "ENABLE_ADAPTIVE_BACKGROUND": False,
                "BACKGROUND_LEARNING_RATE": 0.02,
                "ENABLE_LIGHTING_STABILIZATION": False,
                "LIGHTING_SMOOTH_FACTOR": 0.91,
                "LIGHTING_MEDIAN_WINDOW": 9,
                "MAX_CONTOUR_MULTIPLIER": 33,
                "ENABLE_SIZE_FILTERING": True,
                "MIN_OBJECT_SIZE": 400,
                "MAX_OBJECT_SIZE": 1200,
            }

    monkeypatch.setattr(
        "hydra_suite.trackerkit.gui.dialogs.bg_parameter_helper.BgParameterHelperDialog",
        _FakeDialog,
    )
    monkeypatch.setattr(
        "hydra_suite.trackerkit.gui.orchestrators.config.QMessageBox.information",
        lambda *_args, **_kwargs: None,
    )

    window = _make_main_window(monkeypatch)
    window._setup_panel.file_line.setText(str(video_path))
    window._setup_panel.spin_fps.setValue(20.0)
    window._setup_panel.spin_resize.setValue(0.5)
    window._detection_panel.spin_reference_body_size.setValue(40.0)

    window._config_orch._open_bg_parameter_helper()

    scaled_body_area = math.pi * (40.0 / 2.0) ** 2 * (0.5**2)

    assert window._detection_panel.slider_brightness.value() == 12
    assert window._detection_panel.slider_contrast.value() == 135
    assert window._detection_panel.slider_gamma.value() == 80
    assert window._detection_panel.chk_dark_on_light.isChecked() is False
    assert window._detection_panel.spin_bg_prime.value() == pytest.approx(3.0)
    assert window._detection_panel.chk_adaptive_bg.isChecked() is False
    assert window._detection_panel.spin_bg_learning.value() == pytest.approx(0.02)
    assert window._detection_panel.chk_lighting_stab.isChecked() is False
    assert window._detection_panel.spin_lighting_smooth.value() == pytest.approx(0.91)
    assert window._detection_panel.spin_lighting_median.value() == 9
    assert window._detection_panel.spin_max_contour_multiplier.value() == 33
    assert window._detection_panel.chk_size_filtering.isChecked() is True
    assert window._detection_panel.spin_min_object_size.value() == pytest.approx(
        round(400.0 / scaled_body_area, 2)
    )
    assert window._detection_panel.spin_max_object_size.value() == pytest.approx(
        round(1200.0 / scaled_body_area, 2)
    )
    window.close()
