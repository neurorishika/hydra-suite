from __future__ import annotations

import gc
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtGui = pytest.importorskip("PySide6.QtGui")
QtCore = pytest.importorskip("PySide6.QtCore")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QPixmap = QtGui.QPixmap
QApplication = QtWidgets.QApplication

classkit_config_path = pytest.importorskip(
    "hydra_suite.classkit.gui.project"
).classkit_config_path
classkit_model_dir = pytest.importorskip(
    "hydra_suite.classkit.gui.project"
).classkit_model_dir
classkit_scheme_path = pytest.importorskip(
    "hydra_suite.classkit.gui.project"
).classkit_scheme_path
ClassKitDB = pytest.importorskip("hydra_suite.classkit.core.store.db").ClassKitDB
main_window_module = pytest.importorskip("hydra_suite.classkit.gui.main_window")
MainWindow = main_window_module.MainWindow


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


def test_reset_analysis_view_restores_explore_embedding_mode(qapp) -> None:
    window = MainWindow()
    window.set_explorer_mode("labeling")
    window._show_model_umap = True
    window._show_model_pca = True
    window.btn_umap_model.setEnabled(True)
    window.btn_pca_model.setEnabled(True)
    window.btn_umap_embedding.setChecked(False)
    window.btn_umap_model.setChecked(True)
    window.btn_pca_model.setChecked(True)

    window._reset_analysis_view()

    assert window.explorer_mode == "explore"
    assert window._show_model_umap is False
    assert window._show_model_pca is False
    assert window.btn_umap_embedding.isChecked() is True
    assert window.btn_umap_model.isChecked() is False
    assert window.btn_pca_model.isChecked() is False


def test_outline_control_only_visible_in_predictions_mode(qapp) -> None:
    window = MainWindow()

    window.set_explorer_mode("explore")
    assert window.outline_threshold_label.isHidden() is True
    assert window.outline_threshold_spin.isHidden() is True

    window._model_probs = np.array([[0.9, 0.1]], dtype=np.float32)
    window.set_explorer_mode("predictions")
    assert window.outline_threshold_label.isHidden() is False
    assert window.outline_threshold_spin.isHidden() is False

    window.set_explorer_mode("explore")
    assert window.outline_threshold_label.isHidden() is True
    assert window.outline_threshold_spin.isHidden() is True


def test_labeling_options_only_visible_in_labeling_mode(qapp) -> None:
    window = MainWindow()

    window.set_explorer_mode("explore")
    assert window.labeling_options_group.isHidden() is True

    window.set_explorer_mode("labeling")
    assert window.labeling_options_group.isHidden() is False

    window.set_explorer_mode("explore")
    assert window.labeling_options_group.isHidden() is True


def test_setup_label_shortcuts_ignores_legacy_mode_bindings(qapp) -> None:
    window = MainWindow()
    window.classes = []
    window._custom_shortcuts = {
        "Explore mode": "E",
        "Labeling mode": "L",
        "Review mode": "V",
        "Predictions mode": "P",
        "Approve review label": "A",
    }

    window.setup_label_shortcuts()

    sequences = {shortcut.key().toString() for shortcut in window._label_shortcuts}

    assert "A" in sequences
    assert "E" not in sequences
    assert "L" not in sequences
    assert "V" not in sequences
    assert "P" not in sequences


def test_project_config_sanitizes_removed_mode_shortcuts(qapp) -> None:
    window = MainWindow()

    window._apply_project_config(
        {
            "custom_shortcuts": {
                "Explore mode": "E",
                "Labeling mode": "L",
                "Approve review label": "A",
                "Next unlabeled": "N",
            }
        }
    )

    assert window._custom_shortcuts == {
        "Approve review label": "A",
        "Next unlabeled": "N",
    }


def test_shortcut_help_omits_mode_shortcut_row(qapp) -> None:
    window = MainWindow()

    window._refresh_shortcut_help()

    help_text = window.shortcut_help.text()
    assert "set mode" not in help_text
    assert "approve or reject selected machine label" in help_text


def test_model_projection_buttons_hidden_until_model_loaded(qapp) -> None:
    window = MainWindow()

    assert window.btn_umap_model.isHidden() is True
    assert window.btn_pca_model.isHidden() is True

    window._set_model_projection_buttons_enabled(True)

    assert window.btn_umap_model.isHidden() is False
    assert window.btn_pca_model.isHidden() is False


def test_marker_size_control_updates_explorer(qapp) -> None:
    window = MainWindow()

    window.on_marker_size_changed(1.7)

    assert window._marker_size_multiplier == pytest.approx(1.7)
    assert window.explorer.marker_size_multiplier == pytest.approx(1.7)


def test_review_buttons_live_in_left_review_panel(qapp) -> None:
    window = MainWindow()

    def is_descendant(widget, ancestor) -> bool:
        current = widget
        while current is not None:
            if current is ancestor:
                return True
            current = current.parentWidget()
        return False

    assert is_descendant(window.review_selected_btn, window.context_panel) is True
    assert is_descendant(window.review_reject_btn, window.context_panel) is True
    assert (
        is_descendant(window.review_clear_unverified_btn, window.context_panel) is True
    )

    assert is_descendant(window.review_selected_btn, window.preview_panel) is False
    assert is_descendant(window.review_reject_btn, window.preview_panel) is False
    assert (
        is_descendant(window.review_clear_unverified_btn, window.preview_panel) is False
    )


def test_context_info_label_is_constrained_to_compact_height(qapp) -> None:
    window = MainWindow()

    policy = window.context_info.sizePolicy()

    assert policy.verticalPolicy() == QtWidgets.QSizePolicy.Policy.Maximum
    assert policy.horizontalPolicy() == QtWidgets.QSizePolicy.Policy.Preferred
    assert window.context_info.alignment() & QtCore.Qt.AlignmentFlag.AlignTop


def test_context_panel_renders_compact_project_summary(qapp, tmp_path: Path) -> None:
    project_dir = tmp_path / "head-tail"
    project_dir.mkdir()
    db_path = project_dir / "classkit.db"
    db_path.write_text("db", encoding="utf-8")

    window = MainWindow()
    window.project_path = project_dir
    window.db_path = db_path
    window.image_paths = ["a.png", "b.png", "c.png"]
    window.image_labels = ["left", "", "right"]
    window.classes = ["left", "right", "up", "down"]
    window.embeddings = np.zeros((3, 4), dtype=np.float32)
    window.cluster_assignments = [0, 1, 1]
    window.umap_coords = np.zeros((3, 2), dtype=np.float32)
    window.explorer_mode = "explore"
    window.candidate_indices = []

    window.update_context_panel()

    html = window.context_info.text()
    assert "head-tail" in html
    assert (
        "<td style='color:#8d8d8d; padding:2px 10px 2px 0; vertical-align:top; white-space:nowrap;'>DB</td>"
        in html
    )
    assert "Connected" in html
    assert "Candidates" in html
    assert "none" in html
    assert "Labeling -> Sample next candidates" in html


def test_empty_review_mode_reverts_combo_selection(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown = []

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *args, **kwargs: shown.append(args[1:3]) or QtWidgets.QMessageBox.Ok,
    )

    window = MainWindow()
    review_idx = window.view_mode_combo.findData("review")

    window.view_mode_combo.setCurrentIndex(review_idx)

    assert window.explorer_mode == "explore"
    assert window.view_mode_combo.currentData() == "explore"
    assert shown == [
        ("Review Queue Empty", "There are no unverified machine labels to review yet.")
    ]


def test_first_source_schema_mismatch_can_rewrite_project_schema(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    source_root = tmp_path / "folder_labels"
    ant_train = source_root / "train" / "ant"
    bee_val = source_root / "val" / "bee"
    ant_train.mkdir(parents=True)
    bee_val.mkdir(parents=True)
    (ant_train / "frame001.jpg").write_bytes(b"ant-image")
    (bee_val / "frame002.png").write_bytes(b"bee-image")

    window = MainWindow()
    window.project_path = project_dir
    window.db_path = project_dir / "classkit.db"
    window.classes = ["class_1", "class_2"]

    monkeypatch.setattr(
        MainWindow,
        "_prompt_schema_mismatch_resolution",
        lambda self, source_root, imported_labels, can_rewrite: "rewrite",
    )

    request = window._resolve_ingest_request(source_root)

    assert request == {"source_root": source_root, "import_labels": True}
    assert window.classes == ["ant", "bee"]
    config = json.loads(classkit_config_path(project_dir).read_text(encoding="utf-8"))
    assert config["classes"] == ["ant", "bee"]
    scheme = json.loads(classkit_scheme_path(project_dir).read_text(encoding="utf-8"))
    assert scheme["factors"][0]["labels"] == ["ant", "bee"]


def test_non_first_source_schema_mismatch_can_import_images_only(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hydra_suite.classkit.core.store.db import ClassKitDB

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    db_path = project_dir / "classkit.db"
    db = ClassKitDB(db_path)
    existing_image = project_dir / "existing.jpg"
    existing_image.write_bytes(b"existing")
    db.add_images([existing_image], [None])

    source_root = tmp_path / "folder_labels"
    bee_train = source_root / "train" / "bee"
    bee_train.mkdir(parents=True)
    (bee_train / "frame001.jpg").write_bytes(b"bee-image")

    window = MainWindow()
    window.project_path = project_dir
    window.db_path = db_path
    window.classes = ["ant"]

    monkeypatch.setattr(
        MainWindow,
        "_prompt_schema_mismatch_resolution",
        lambda self, source_root, imported_labels, can_rewrite: "images_only",
    )

    request = window._resolve_ingest_request(source_root)

    assert request == {"source_root": source_root, "import_labels": False}


def test_open_class_editor_clears_labels_removed_from_new_scheme(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    db = ClassKitDB(project_dir / "classkit.db")

    image_a = project_dir / "a.jpg"
    image_b = project_dir / "b.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    db.add_images([image_a, image_b])
    db.update_labels_batch(
        {
            str(image_a.resolve()): "red|left",
            str(image_b.resolve()): "blue|right",
        }
    )

    class FakeDialog:
        Accepted = 1

        def __init__(self, *args, **kwargs) -> None:
            self.flat_classes = ["red"]

        def exec(self) -> int:
            return self.Accepted

        def get_scheme_dict(self) -> dict:
            return {
                "name": "directional",
                "description": "updated",
                "factors": [
                    {"name": "color", "labels": ["red"], "shortcut_keys": []},
                    {
                        "name": "side",
                        "labels": ["left", "right"],
                        "shortcut_keys": [],
                    },
                ],
                "training_modes": ["flat_custom", "multihead_custom"],
            }

    monkeypatch.setattr(
        "hydra_suite.classkit.gui.dialogs.ClassEditorDialog",
        FakeDialog,
    )

    window = MainWindow()
    window.project_path = project_dir
    window.db_path = db.db_path
    window.image_paths = [image_a.resolve(), image_b.resolve()]
    window.image_labels = db.get_all_labels()

    monkeypatch.setattr(window, "rebuild_label_buttons", lambda: None)
    monkeypatch.setattr(window, "setup_label_shortcuts", lambda: None)
    monkeypatch.setattr(window, "update_context_panel", lambda: None)
    monkeypatch.setattr(window, "update_explorer_plot", lambda *args, **kwargs: None)

    window.open_class_editor()

    assert db.get_all_labels() == ["red|left", None]
    assert window.image_labels == ["red|left", None]
    assert window.classes == ["red"]


def test_validate_training_pairs_prunes_labels_removed_from_active_scheme(
    qapp, tmp_path: Path
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    db = ClassKitDB(project_dir / "classkit.db")

    image_paths = []
    for index, label in enumerate(
        ["left", "right", "left", "right", "legacy"],
        start=1,
    ):
        image_path = project_dir / f"img_{index}.jpg"
        image_path.write_bytes(label.encode("utf-8"))
        image_paths.append(image_path)
    db.add_images(image_paths)
    db.update_labels_batch(
        {
            str(path.resolve()): label
            for path, label in zip(
                image_paths,
                ["left", "right", "left", "right", "legacy"],
            )
        }
    )

    scheme_path = classkit_scheme_path(project_dir)
    scheme_path.parent.mkdir(parents=True, exist_ok=True)
    scheme_path.write_text(
        json.dumps(
            {
                "name": "direction",
                "description": "only active labels",
                "factors": [
                    {
                        "name": "direction",
                        "labels": ["left", "right"],
                        "shortcut_keys": [],
                    }
                ],
                "training_modes": ["flat_custom", "flat_yolo"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    window = MainWindow()
    window.project_path = project_dir
    window.db_path = db.db_path
    window.image_paths = [path.resolve() for path in image_paths]
    window.image_labels = db.get_all_labels()
    window.embeddings = np.zeros((len(image_paths), 4), dtype=np.float32)

    labeled_pairs = window._validate_training_pairs()

    assert labeled_pairs is not None
    assert [label for _, label in labeled_pairs] == ["left", "right", "left", "right"]
    assert db.get_all_labels() == ["left", "right", "left", "right", None]
    assert window.image_labels == ["left", "right", "left", "right", None]


def test_make_training_spec_propagates_custom_finetune_settings(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    settings = {
        "custom_backbone": "resnet18",
        "custom_fine_tune_method": "gradual_unfreeze",
        "custom_trainable_layers": 3,
        "custom_backbone_lr_scale": 0.2,
        "custom_layerwise_lr_decay": 0.55,
        "custom_gradual_unfreeze_interval": 7,
        "custom_input_size": 192,
        "epochs": 12,
        "batch": 8,
        "lr": 0.0007,
        "patience": 4,
    }

    spec = window._make_training_spec(
        settings,
        window._training_role_for_mode("flat_custom"),
        "flat_custom",
        False,
        tmp_path / "export",
    )

    assert spec.custom_params is not None
    assert spec.custom_params.backbone == "resnet18"
    assert spec.custom_params.fine_tune_method == "gradual_unfreeze"
    assert spec.custom_params.trainable_layers == 3
    assert spec.custom_params.backbone_lr_scale == pytest.approx(0.2)
    assert spec.custom_params.layerwise_lr_decay == pytest.approx(0.55)
    assert spec.custom_params.gradual_unfreeze_interval == 7


def test_labeled_eval_arrays_prefers_active_evaluation_subset(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    image_paths = [tmp_path / f"img_{idx}.png" for idx in range(4)]
    window.image_paths = [str(path) for path in image_paths]
    window.image_labels = ["left", "left", "right", "right"]
    window.classes = ["left", "right"]
    window._set_active_evaluation_selection(
        [str(image_paths[1]), str(image_paths[2])],
        "val",
    )

    idx_arr, y_true, scope_text = window._labeled_eval_arrays()

    assert idx_arr.tolist() == [1, 2]
    assert y_true.tolist() == [0, 1]
    assert scope_text == "Val split"


def test_set_active_evaluation_from_meta_restores_split_memberships(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    train_path = tmp_path / "train.png"
    val_path = tmp_path / "val.png"
    test_path = tmp_path / "test.png"

    window._set_active_evaluation_from_meta(
        {
            "evaluation": {
                "split_name": "test",
                "paths": [str(test_path)],
                "split_paths": {
                    "train": [str(train_path)],
                    "val": [str(val_path)],
                    "test": [str(test_path)],
                },
            }
        }
    )

    assert window._active_evaluation_split_name == "test"
    assert window._active_evaluation_paths == {str(test_path.resolve())}
    assert window._split_membership_paths == {
        "train": {str(train_path.resolve())},
        "val": {str(val_path.resolve())},
        "test": {str(test_path.resolve())},
    }


def test_labeled_eval_arrays_include_labels_outside_current_scheme(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    image_paths = [tmp_path / f"img_{idx}.png" for idx in range(3)]
    window.image_paths = [str(path) for path in image_paths]
    window.image_labels = ["left", "unknown", "up"]
    window.classes = ["left", "right"]

    idx_arr, y_true, scope_text = window._labeled_eval_arrays()

    assert window._evaluation_class_names() == ["left", "right", "unknown", "up"]
    assert idx_arr.tolist() == [0, 1, 2]
    assert y_true.tolist() == [0, 2, 3]
    assert scope_text == "All labeled project images"


def test_export_dataset_includes_labels_outside_current_scheme(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class FakeDialog:
        def __init__(self, default_output: str = "", parent=None) -> None:
            captured["default_output"] = default_output

        def exec(self) -> bool:
            return True

        def get_settings(self) -> dict[str, object]:
            return {
                "output_dir": str(tmp_path / "exports"),
                "format": "imagefolder",
                "copy_files": True,
                "include_unlabeled": False,
                "include_unverified_labels": False,
                "val_fraction": 0.2,
                "test_fraction": 0.0,
            }

    class FakeWorker:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.signals = type(
                "Signals",
                (),
                {
                    "started": type(
                        "Signal", (), {"connect": staticmethod(lambda _fn: None)}
                    )(),
                    "progress": type(
                        "Signal", (), {"connect": staticmethod(lambda _fn: None)}
                    )(),
                    "success": type(
                        "Signal", (), {"connect": staticmethod(lambda _fn: None)}
                    )(),
                    "error": type(
                        "Signal", (), {"connect": staticmethod(lambda _fn: None)}
                    )(),
                    "finished": type(
                        "Signal", (), {"connect": staticmethod(lambda _fn: None)}
                    )(),
                },
            )()

    monkeypatch.setattr(
        "hydra_suite.classkit.gui.dialogs.ExportDialog",
        FakeDialog,
    )
    monkeypatch.setattr(
        "hydra_suite.classkit.jobs.task_workers.ExportWorker",
        FakeWorker,
    )

    window = MainWindow()
    window.project_path = tmp_path / "project"
    window.project_path.mkdir()
    window.image_paths = [tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"]
    window.image_labels = ["left", "unknown", None]
    window.classes = ["left", "right", "up", "down"]
    window._flush_pending_label_updates = lambda force=False: None
    window._threadpool_start = lambda worker: captured.setdefault("worker", worker)

    window.export_dataset()

    assert captured["image_paths"] == [tmp_path / "a.png", tmp_path / "b.png"]
    assert captured["labels"] == [0, 4]
    assert captured["class_names"] == {
        0: "left",
        1: "right",
        2: "up",
        3: "down",
        4: "unknown",
    }


def test_assign_label_can_continue_in_infinite_labeling_mode(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_single_shot = main_window_module.QTimer.singleShot

    monkeypatch.setattr(
        main_window_module.QTimer,
        "singleShot",
        staticmethod(
            lambda msec, fn: fn() if int(msec) == 0 else original_single_shot(msec, fn)
        ),
    )

    window = MainWindow()
    window.image_paths = [
        Path("/tmp/a.png"),
        Path("/tmp/b.png"),
        Path("/tmp/c.png"),
        Path("/tmp/d.png"),
    ]
    window.image_labels = [None, None, None, None]
    window.embeddings = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.5, 10.2],
        ],
        dtype=np.float32,
    )
    window.cluster_assignments = np.array([0, 0, 1, 1])
    window.set_explorer_mode("labeling")
    window.selected_point_index = 0
    window.candidate_indices = []

    monkeypatch.setattr(window, "_prompt_after_label_set_complete", lambda: "infinite")

    window.assign_label_to_selected("alpha")

    assert window.image_labels[0] == "alpha"
    assert window._labeling_flow_mode == "infinite"
    assert window.selected_point_index in {2, 3}
    assert window.candidate_indices == [window.selected_point_index]
    assert window.round_labeled_indices == []
    assert int(window.cluster_assignments[window.selected_point_index]) == 1


def test_manual_click_in_labeling_mode_browses_database_with_prev_next(qapp) -> None:
    window = MainWindow()
    window.image_paths = [Path(f"/tmp/{idx}.png") for idx in range(6)]
    window.image_labels = [None] * 6
    window.candidate_indices = [1, 4]
    window.set_explorer_mode("labeling")

    window.on_explorer_point_clicked(2)
    window.on_next_image()

    assert window.selected_point_index == 3

    window._command_busy = False
    window._command_block_until = 0.0
    window.on_prev_image()

    assert window.selected_point_index == 2


def test_clicking_different_point_replaces_labeling_selection(qapp) -> None:
    window = MainWindow()
    window.image_paths = [Path(f"/tmp/{idx}.png") for idx in range(5)]
    window.image_labels = [None] * 5
    window.candidate_indices = [1, 3]
    window.set_explorer_mode("labeling")

    window.on_explorer_point_clicked(1)
    window.on_explorer_point_clicked(4)

    assert window.selected_point_index == 4
    assert window._labeling_navigation_scope == "database"


def test_infinite_labeling_reuses_cached_distance_scores(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_single_shot = main_window_module.QTimer.singleShot

    monkeypatch.setattr(
        main_window_module.QTimer,
        "singleShot",
        staticmethod(
            lambda msec, fn: fn() if int(msec) == 0 else original_single_shot(msec, fn)
        ),
    )

    window = MainWindow()
    window.image_paths = [
        Path("/tmp/a.png"),
        Path("/tmp/b.png"),
        Path("/tmp/c.png"),
        Path("/tmp/d.png"),
    ]
    window.image_labels = ["seed", None, None, None]
    window.embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [8.0, 8.0],
            [8.5, 8.2],
        ],
        dtype=np.float32,
    )
    window.cluster_assignments = np.array([0, 0, 1, 1])

    assert window._start_infinite_labeling_mode() is True
    first_choice = window.selected_point_index
    assert first_choice in {2, 3}

    monkeypatch.setattr(
        window,
        "_min_distance_to_reference_scores",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("full distance recomputation should not run again")
        ),
    )

    window.assign_label_to_selected("beta")

    assert window._labeling_flow_mode == "infinite"
    assert window.selected_point_index in {1, 2, 3}
    assert window.selected_point_index != first_choice


def test_infinite_labeling_restores_persisted_cache_before_recomputing(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_module = pytest.importorskip("hydra_suite.classkit.core.store.db")

    class FakeDB:
        def __init__(self, _path):
            self.path = _path

        def get_most_recent_infinite_label_cache(self, embedding_cache_id=None):
            assert embedding_cache_id == 41
            return {
                "distance_cache": np.array([0.0, 0.25, 81.0], dtype=np.float32),
                "cluster_counts": np.array([1, 0], dtype=np.int32),
                "owner_cache": np.array([0, 0, 0], dtype=np.int32),
                "embedding_cache_id": 41,
                "cluster_cache_id": 9,
                "label_signature": window._labeled_index_signature([0]),
            }

    window = MainWindow()
    window.db_path = tmp_path / "classkit.db"
    window.image_paths = [Path("/tmp/a.png"), Path("/tmp/b.png"), Path("/tmp/c.png")]
    window.image_labels = ["seed", None, None]
    window.embeddings = np.array([[0.0, 0.0], [0.5, 0.0], [9.0, 9.0]], dtype=np.float32)
    window.cluster_assignments = np.array([0, 0, 1], dtype=np.int32)
    window._current_embedding_cache_id = 41
    window._current_cluster_cache_id = 9

    monkeypatch.setattr(db_module, "ClassKitDB", FakeDB)
    monkeypatch.setattr(
        window,
        "_min_distance_to_reference_scores",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("distance cache should restore from project storage")
        ),
    )

    restored = window._ensure_infinite_labeling_cache()

    assert restored is not None
    assert np.array_equal(
        restored,
        np.array([0.0, 0.25, 81.0], dtype=np.float32),
    )
    assert np.array_equal(
        window._infinite_label_cluster_counts,
        np.array([1, 0], dtype=np.int32),
    )
    assert np.array_equal(
        window._infinite_label_owner_cache,
        np.array([0, 0, 0], dtype=np.int32),
    )


def test_deleting_label_keeps_infinite_cache_hot_for_next_label(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_single_shot = main_window_module.QTimer.singleShot

    monkeypatch.setattr(
        main_window_module.QTimer,
        "singleShot",
        staticmethod(
            lambda msec, fn: fn() if int(msec) == 0 else original_single_shot(msec, fn)
        ),
    )

    window = MainWindow()
    window.image_paths = [Path(f"/tmp/{idx}.png") for idx in range(5)]
    window.image_labels = ["alpha", "beta", None, None, None]
    window.embeddings = np.array(
        [
            [0.0, 0.0],
            [0.4, 0.0],
            [7.0, 7.0],
            [7.4, 7.1],
            [0.8, 0.1],
        ],
        dtype=np.float32,
    )
    window.cluster_assignments = np.array([0, 0, 1, 1, 0], dtype=np.int32)
    window.set_explorer_mode("labeling")
    window._labeling_flow_mode = "infinite"
    window.selected_point_index = 2
    window.candidate_indices = [2]

    assert window._ensure_infinite_labeling_cache() is not None
    assert window._infinite_label_owner_cache is not None

    window._set_label_for_index(1, None)

    assert window._infinite_label_distance_cache is not None
    assert window._infinite_label_owner_cache is not None

    monkeypatch.setattr(
        window,
        "_min_distance_to_reference_scores",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("full cache rebuild should not run after delete")
        ),
    )
    monkeypatch.setattr(
        window,
        "_min_distance_and_owner_to_reference_scores",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("owner-aware full cache rebuild should not run after delete")
        ),
    )

    window.assign_label_to_selected("gamma")

    assert window.image_labels[2] == "gamma"
    assert window._labeling_flow_mode == "infinite"
    assert window.selected_point_index in {3, 4}


def test_after_training_inference_skips_invalid_auto_model_umap(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    critical_calls = []

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        lambda *args, **kwargs: critical_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        main_window_module.QTimer,
        "singleShot",
        staticmethod(lambda _msec, fn: fn()),
    )

    class Dialog:
        class ProgressBar:
            def __init__(self) -> None:
                self.value = None

            def setValue(self, value: int) -> None:
                self.value = value

        def __init__(self) -> None:
            self.logs = []
            self.progress_bar = self.ProgressBar()

        def append_log(self, message: str) -> None:
            self.logs.append(message)

    window = MainWindow()
    window._model_probs = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    window.tabs.setCurrentIndex(0)
    dialog = Dialog()

    window._after_training_inference(dialog)

    assert critical_calls == []
    assert window.tabs.currentWidget() is window.metrics_page
    assert dialog.progress_bar.value == 100
    assert dialog.logs == [
        "Auto-computing model-space UMAP...",
        "Model-space UMAP skipped: Need at least 2 prediction columns to compute model-space UMAP.",
        "Training Complete. Metrics tab updated.",
    ]


def test_after_training_inference_logs_model_umap_worker_error_without_modal(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    critical_calls = []

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        lambda *args, **kwargs: critical_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        main_window_module.QTimer,
        "singleShot",
        staticmethod(lambda _msec, fn: fn()),
    )

    task_workers = pytest.importorskip("hydra_suite.classkit.jobs.task_workers")

    class DummySignal:
        def __init__(self) -> None:
            self._callbacks = []

        def connect(self, callback) -> None:
            self._callbacks.append(callback)

        def emit(self, *args) -> None:
            for callback in list(self._callbacks):
                callback(*args)

    class DummySignals:
        def __init__(self) -> None:
            self.success = DummySignal()
            self.error = DummySignal()
            self.progress = DummySignal()
            self.finished = DummySignal()

    class FailingWorker:
        def __init__(self, *args, **kwargs) -> None:
            self.signals = DummySignals()

    class Dialog:
        class ProgressBar:
            def __init__(self) -> None:
                self.value = None

            def setValue(self, value: int) -> None:
                self.value = value

        def __init__(self) -> None:
            self.logs = []
            self.progress_bar = self.ProgressBar()

        def append_log(self, message: str) -> None:
            self.logs.append(message)

    monkeypatch.setattr(task_workers, "LogitsUMAPWorker", FailingWorker)

    window = MainWindow()
    window._model_probs = np.array(
        [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]],
        dtype=np.float32,
    )
    window._threadpool_start = lambda worker: (
        worker.signals.error.emit("boom"),
        worker.signals.finished.emit(),
    )
    dialog = Dialog()

    window._after_training_inference(dialog)

    assert critical_calls == []
    assert dialog.progress_bar.value == 100
    assert dialog.logs == [
        "Auto-computing model-space UMAP...",
        "Model-space UMAP failed: boom",
        "Training Complete. Metrics tab updated.",
    ]


def test_empty_hover_clears_preview_without_active_label_selection(qapp) -> None:
    window = MainWindow()
    window.set_explorer_mode("labeling")
    window.selected_point_index = None
    window.hover_locked = False
    window.last_preview_index = 5
    window.preview_canvas.set_pixmap(QPixmap(10, 10))

    window.on_explorer_empty_hover()

    assert window.last_preview_index is None
    assert window.preview_canvas.pix_item.pixmap().isNull() is True
    assert "Hover a point to preview the source image" in window.preview_info.text()


def test_empty_hover_clears_preview_in_explore_mode_and_cancels_pending_hover(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    window.set_explorer_mode("explore")
    window.selected_point_index = None
    window.hover_locked = False

    image_path = tmp_path / "preview.png"
    pixmap = QPixmap(10, 10)
    assert pixmap.save(str(image_path)) is True

    window.image_paths = [image_path]
    window.request_preview_for_index(0, source="hover")

    window.on_explorer_empty_hover()
    window._flush_preview_update()

    assert window.last_preview_index is None
    assert window._pending_preview_index is None
    assert window.preview_canvas.pix_item.pixmap().isNull() is True
    assert "Hover a point to preview the source image" in window.preview_info.text()


def test_empty_hover_keeps_selected_preview_with_locked_selection(qapp) -> None:
    window = MainWindow()
    window.set_explorer_mode("labeling")
    window.selected_point_index = 3
    window.hover_locked = True
    window.last_preview_index = 3
    window.preview_canvas.set_pixmap(QPixmap(10, 10))

    window.on_explorer_empty_hover()

    assert window._pending_preview_index == 3
    assert window._pending_preview_source == "selection"
    assert window.last_preview_index == 3
    assert window.preview_canvas.pix_item.pixmap().isNull() is False


def test_hovering_other_points_does_not_replace_locked_labeling_preview(qapp) -> None:
    window = MainWindow()
    window.set_explorer_mode("labeling")
    window.selected_point_index = 2
    window.hover_locked = True
    window.candidate_indices = [1, 2, 3]

    requested = []
    window.request_preview_for_index = lambda index, source="hover": requested.append(
        (index, source)
    )

    window.on_explorer_point_hovered(1)

    assert requested == []


def test_empty_background_double_click_clears_labeling_pool_but_stays_in_labeling(
    qapp,
) -> None:
    window = MainWindow()
    window.image_paths = [Path(f"/tmp/{idx}.png") for idx in range(4)]
    window.image_labels = [None] * 4
    window.set_explorer_mode("labeling")
    window.selected_point_index = 2
    window.hover_locked = True
    window.candidate_indices = [1, 2]
    window.round_labeled_indices = [1]
    window._labeling_flow_mode = "infinite"
    window._labeling_navigation_scope = "database"

    window.on_explorer_background_double_click()

    assert window.explorer_mode == "labeling"
    assert window.selected_point_index is None
    assert window.hover_locked is False
    assert window.candidate_indices == []
    assert window.round_labeled_indices == []
    assert window._labeling_flow_mode == "batch"
    assert window._labeling_navigation_scope == "pool"


def test_empty_background_double_click_keeps_labeling_mode_without_active_set(
    qapp,
) -> None:
    window = MainWindow()
    window.image_paths = [Path(f"/tmp/{idx}.png") for idx in range(3)]
    window.image_labels = [None] * 3
    window.set_explorer_mode("labeling")
    window.selected_point_index = 1
    window.hover_locked = True

    window.on_explorer_background_double_click()

    assert window.explorer_mode == "labeling"
    assert window.selected_point_index is None
    assert window.hover_locked is False


def test_leaving_labeling_mode_clears_selection_and_restores_hover_preview(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    window.set_explorer_mode("labeling")
    window.selected_point_index = 0
    window.hover_locked = True
    window.last_preview_index = 0
    window.preview_canvas.set_pixmap(QPixmap(10, 10))

    image_a = tmp_path / "labeling_a.png"
    image_b = tmp_path / "labeling_b.png"
    pixmap = QPixmap(10, 10)
    assert pixmap.save(str(image_a)) is True
    assert pixmap.save(str(image_b)) is True
    window.image_paths = [image_a, image_b]

    window.set_explorer_mode("explore")

    assert window.selected_point_index is None
    assert window.hover_locked is False
    assert window.preview_canvas.pix_item.pixmap().isNull() is True
    assert "selection lock is disabled in this mode" in window.selection_info.text()

    window.on_explorer_point_hovered(1)
    window._flush_preview_update()

    assert window.last_preview_index == 1


def test_leaving_review_mode_clears_selection_and_restores_hover_preview(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    window._review_candidate_indices = [0]
    window.selected_point_index = 0
    window.hover_locked = True
    window.last_preview_index = 0
    window.preview_canvas.set_pixmap(QPixmap(10, 10))
    window.set_explorer_mode("review")

    image_a = tmp_path / "review_a.png"
    image_b = tmp_path / "review_b.png"
    pixmap = QPixmap(10, 10)
    assert pixmap.save(str(image_a)) is True
    assert pixmap.save(str(image_b)) is True
    window.image_paths = [image_a, image_b]

    window.set_explorer_mode("explore")

    assert window.selected_point_index is None
    assert window.hover_locked is False
    assert window.preview_canvas.pix_item.pixmap().isNull() is True

    window.on_explorer_point_hovered(1)
    window._flush_preview_update()

    assert window.last_preview_index == 1


def test_preview_panel_shows_prediction_details(qapp, tmp_path: Path) -> None:
    from hydra_suite.classkit.gui.widgets.color_utils import to_hex

    window = MainWindow()
    image_path = tmp_path / "prediction_preview.png"
    image_path.write_bytes(b"image-bytes")

    window.image_paths = [image_path]
    window.image_labels = ["reviewed"]
    window.cluster_assignments = np.array([4], dtype=np.int32)
    window._model_class_names = ["zebra", "ant", "bee"]
    window._model_probs = np.array([[0.15, 0.80, 0.05]], dtype=np.float32)
    window.classes = ["ant", "zebra"]

    window.load_preview_for_index(0)

    preview_html = window.preview_info.text()
    selection_html = window.selection_info.text()
    assert "Prediction:" in preview_html
    assert "ant" in preview_html
    assert "80.0%" in preview_html
    assert "Top-3:" in preview_html
    assert "Prediction:" in selection_html
    assert "ant" in selection_html
    expected_color = to_hex(
        window._schema_category_color_map(extra_categories=["zebra", "ant", "bee"])[
            "ant"
        ]
    )
    assert f"background-color:{expected_color}" in preview_html


def test_preview_panel_shows_multihead_prediction_details(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    image_path = tmp_path / "prediction_preview_multihead.png"
    image_path.write_bytes(b"image-bytes")

    window.image_paths = [image_path]
    window.image_labels = ["blue|right"]
    window.cluster_assignments = np.array([2], dtype=np.int32)
    window.classes = ["blue", "red"]
    window._set_model_prediction_state(
        np.array([[0.15, 0.85, 0.20, 0.80]], dtype=np.float32),
        ["blue", "red", "left", "right"],
        active_model_mode="custom_cnn_multihead",
        prediction_heads=[
            {"factor": "color", "class_names": ["blue", "red"]},
            {"factor": "side", "class_names": ["left", "right"]},
        ],
    )

    window.load_preview_for_index(0)

    preview_html = window.preview_info.text()
    selection_html = window.selection_info.text()
    assert "Composite Prediction:" in preview_html
    assert "red|right" in preview_html
    assert "color:" in preview_html
    assert "side:" in preview_html
    assert "85.0%" in preview_html
    assert "80.0%" in preview_html
    assert "Composite Prediction:" in selection_html


def test_prediction_mode_uses_composite_multihead_labels(qapp) -> None:
    from hydra_suite.classkit.gui.widgets.color_utils import build_category_color_map

    window = MainWindow()
    window.image_paths = [Path("/tmp/a.png"), Path("/tmp/b.png")]
    window.image_labels = [None, None]
    window.classes = ["blue", "red"]
    window.cluster_assignments = np.array([0, 1], dtype=np.int32)
    window.umap_coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    window._set_model_prediction_state(
        np.array(
            [
                [0.10, 0.90, 0.80, 0.20],
                [0.70, 0.30, 0.25, 0.75],
            ],
            dtype=np.float32,
        ),
        ["blue", "red", "left", "right"],
        active_model_mode="custom_cnn_multihead",
        prediction_heads=[
            {"factor": "color", "class_names": ["blue", "red"]},
            {"factor": "side", "class_names": ["left", "right"]},
        ],
    )

    assert window._prediction_labels_for_plot() == ["red|left", "blue|right"]

    window.set_explorer_mode("predictions")
    window.update_explorer_plot(force_fit=True)

    expected_map = build_category_color_map(
        ["blue", "red", "unknown", "red|left", "blue|right"],
        category_order=["blue", "red", "unknown", "red|left", "blue|right"],
    )
    point_colors = [item.brush().color().name() for item in window.explorer.points]

    assert point_colors[0] == expected_map["red|left"].name()
    assert point_colors[1] == expected_map["blue|right"].name()
    assert "By factor:" in window.explorer.points[0].toolTip()
    assert "color: red" in window.explorer.points[0].toolTip()
    assert "side: left" in window.explorer.points[0].toolTip()


def test_prediction_mode_uses_schema_map_colors(qapp) -> None:
    from hydra_suite.classkit.gui.widgets.color_utils import build_category_color_map

    window = MainWindow()
    window.image_paths = [Path("/tmp/a.png"), Path("/tmp/b.png")]
    window.image_labels = [None, None]
    window.classes = ["ant", "zebra"]
    window.cluster_assignments = np.array([0, 1], dtype=np.int32)
    window.umap_coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    window._model_class_names = ["zebra", "ant"]
    window._model_probs = np.array([[0.05, 0.95], [0.99, 0.01]], dtype=np.float32)

    window.set_explorer_mode("predictions")
    window.update_explorer_plot(force_fit=True)

    expected_map = build_category_color_map(
        ["ant", "zebra", "unknown"],
        category_order=["ant", "zebra", "unknown"],
    )
    point_colors = [item.brush().color().name() for item in window.explorer.points]

    assert point_colors[0] == expected_map["ant"].name()
    assert point_colors[1] == expected_map["zebra"].name()


def test_prediction_mode_zoom_preserves_schema_map_colors(qapp) -> None:
    from PySide6.QtCore import QPoint

    from hydra_suite.classkit.gui.widgets.color_utils import build_category_color_map

    class _WheelEvent:
        def __init__(self, delta: int) -> None:
            self._delta = delta

        def angleDelta(self):
            return QPoint(0, self._delta)

    window = MainWindow()
    window.image_paths = [Path("/tmp/a.png"), Path("/tmp/b.png")]
    window.image_labels = [None, None]
    window.classes = ["ant", "zebra"]
    window.cluster_assignments = np.array([0, 1], dtype=np.int32)
    window.umap_coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    window._model_class_names = ["zebra", "ant"]
    window._model_probs = np.array([[0.05, 0.95], [0.99, 0.01]], dtype=np.float32)

    window.set_explorer_mode("predictions")
    window.update_explorer_plot(force_fit=True)

    expected_map = build_category_color_map(
        ["ant", "zebra", "unknown"],
        category_order=["ant", "zebra", "unknown"],
    )
    before_colors = [item.brush().color().name() for item in window.explorer.points]

    window.explorer.wheelEvent(_WheelEvent(120))

    after_colors = [item.brush().color().name() for item in window.explorer.points]

    assert before_colors == after_colors
    assert after_colors[0] == expected_map["ant"].name()
    assert after_colors[1] == expected_map["zebra"].name()


def test_prediction_tooltip_only_exists_in_prediction_mode(qapp) -> None:
    window = MainWindow()
    window.image_paths = [Path("/tmp/a.png")]
    window.image_labels = [None]
    window.cluster_assignments = np.array([0], dtype=np.int32)
    window.umap_coords = np.array([[0.0, 0.0]], dtype=np.float32)
    window._model_class_names = ["zebra", "ant"]
    window._model_probs = np.array([[0.1, 0.9]], dtype=np.float32)

    window.set_explorer_mode("predictions")
    window.update_explorer_plot(force_fit=True)

    assert "Prediction: ant (90.0%)" in window.explorer.points[0].toolTip()
    assert "Top predictions:" in window.explorer.points[0].toolTip()

    window.set_explorer_mode("explore")
    window.update_explorer_plot(force_fit=True)

    assert window.explorer.points[0].toolTip() == ""


def test_training_settings_persist_in_project_config(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    window.project_path = tmp_path
    window._last_training_settings = {
        "mode": "flat_custom",
        "custom_input_size": 192,
        "tiny_preset": "large",
        "tiny_width": 160,
        "tiny_height": 96,
        "device": "cpu",
    }

    window._save_last_training_settings()

    config = json.loads(classkit_config_path(tmp_path).read_text())
    assert config["last_training_settings"]["custom_input_size"] == 192

    restored = MainWindow()
    restored._apply_project_config(config)

    assert restored._last_training_settings["mode"] == "flat_custom"
    assert restored._last_training_settings["tiny_preset"] == "large"
    assert restored._last_training_settings["tiny_width"] == 160


def test_default_training_settings_use_average_image_dimensions(
    qapp, tmp_path: Path
) -> None:
    from PIL import Image

    image_a = tmp_path / "img_a.png"
    image_b = tmp_path / "img_b.png"
    Image.new("RGB", (104, 60), color=(32, 64, 96)).save(image_a)
    Image.new("RGB", (136, 68), color=(96, 64, 32)).save(image_b)

    window = MainWindow()
    window.image_paths = [image_a, image_b]

    defaults = window._default_training_settings_from_project()

    assert defaults["tiny_width"] == 128
    assert defaults["tiny_height"] == 64
    assert defaults["custom_input_size"] == 96


def test_make_training_spec_uses_selected_initial_model_path(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()

    yolo_start = tmp_path / "previous_yolo.pt"
    yolo_start.write_text("weights", encoding="utf-8")
    yolo_spec = window._make_training_spec(
        {
            "base_model": "yolo26n-cls.pt",
            "initial_model_path": str(yolo_start),
            "epochs": 5,
            "batch": 8,
            "lr": 0.001,
            "patience": 2,
        },
        window._training_role_for_mode("flat_yolo"),
        "flat_yolo",
        True,
        tmp_path / "export_yolo",
    )

    assert yolo_spec.base_model == str(yolo_start)
    assert yolo_spec.resume_from == ""

    custom_start = tmp_path / "previous_custom.pth"
    custom_start.write_text("weights", encoding="utf-8")
    custom_spec = window._make_training_spec(
        {
            "custom_backbone": "resnet18",
            "initial_model_path": str(custom_start),
            "epochs": 5,
            "batch": 8,
            "lr": 0.001,
            "patience": 2,
        },
        window._training_role_for_mode("flat_custom"),
        "flat_custom",
        False,
        tmp_path / "export_custom",
    )

    assert custom_spec.base_model == ""
    assert custom_spec.resume_from == str(custom_start)


def test_make_training_spec_propagates_tiny_preset(qapp, tmp_path: Path) -> None:
    window = MainWindow()

    tiny_spec = window._make_training_spec(
        {
            "tiny_preset": "large",
            "tiny_dim": 128,
            "tiny_dropout": 0.05,
        },
        window._training_role_for_mode("flat_custom"),
        "flat_custom",
        False,
        tmp_path / "export_custom",
    )

    assert tiny_spec.tiny_params.tiny_preset == "large"
    assert tiny_spec.custom_params is not None
    assert tiny_spec.custom_params.tiny_preset == "large"
    assert tiny_spec.custom_params.hidden_dim == 128
    assert tiny_spec.custom_params.dropout == pytest.approx(0.05)


def test_make_training_spec_maps_color_augmentations(qapp, tmp_path: Path) -> None:
    window = MainWindow()

    spec = window._make_training_spec(
        {
            "base_model": "yolo26n-cls.pt",
            "epochs": 5,
            "batch": 8,
            "lr": 0.001,
            "patience": 2,
            "hue": 0.04,
            "saturation": 0.35,
            "brightness": 0.2,
            "contrast": 0.15,
            "monochrome": True,
        },
        window._training_role_for_mode("flat_yolo"),
        "flat_yolo",
        True,
        tmp_path / "export_yolo",
    )

    assert spec.augmentation_profile.hue == pytest.approx(0.04)
    assert spec.augmentation_profile.saturation == pytest.approx(0.35)
    assert spec.augmentation_profile.brightness == pytest.approx(0.2)
    assert spec.augmentation_profile.contrast == pytest.approx(0.15)
    assert spec.augmentation_profile.monochrome is True
    assert spec.augmentation_profile.args["hsv_h"] == pytest.approx(0.04)
    assert spec.augmentation_profile.args["hsv_s"] == pytest.approx(0.35)
    assert spec.augmentation_profile.args["hsv_v"] == pytest.approx(0.2)
    assert "contrast" not in spec.augmentation_profile.args


def test_autoload_cached_analysis_restores_state_without_prompt(qapp) -> None:
    class FakeDB:
        def get_most_recent_embeddings(self):
            return np.array([[1.0, 2.0], [3.0, 4.0]]), {"id": 17}

        def get_most_recent_cluster_cache(self, embedding_cache_id=None):
            assert embedding_cache_id == 17
            return {"assignments": np.array([0, 1]), "n_clusters": 2}

        def get_most_recent_umap_cache(self, embedding_cache_id=None, kind="embedding"):
            assert embedding_cache_id == 17
            assert kind == "embedding"
            return {
                "coords": np.array([[0.1, 0.2], [0.3, 0.4]]),
                "n_neighbors": 21,
                "min_dist": 0.05,
            }

        def get_most_recent_candidate_cache(self):
            return {"candidate_indices": [0, 1]}

    window = MainWindow()
    window.image_paths = [Path("/tmp/a.png"), Path("/tmp/b.png")]
    window.image_labels = ["kept-label", None]
    window._ask_yes_no = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("autoload should not prompt")
    )

    window._autoload_cached_analysis(FakeDB())

    assert window._current_embedding_cache_id == 17
    assert np.array_equal(window.embeddings, np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.array_equal(window.cluster_assignments, np.array([0, 1]))
    assert np.array_equal(window.umap_coords, np.array([[0.1, 0.2], [0.3, 0.4]]))
    assert window.last_umap_params == {"n_neighbors": 21, "min_dist": 0.05}
    assert window.candidate_indices == [0, 1]
    assert window.round_labeled_indices == [0]
    assert window.explorer_mode == "labeling"


def test_restore_cached_candidate_batch_marks_existing_labels(qapp) -> None:
    window = MainWindow()
    window.image_paths = [
        Path("/tmp/a.png"),
        Path("/tmp/b.png"),
        Path("/tmp/c.png"),
    ]
    window.image_labels = ["alpha", None, "beta"]

    window._restore_cached_candidate_batch([0, 1, 2, 1, -1, 99, "2"])

    assert window.candidate_indices == [0, 1, 2]
    assert window.round_labeled_indices == [0, 2]


def test_autoload_model_from_db_prefers_cached_predictions_without_prompt(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_module = pytest.importorskip("hydra_suite.classkit.core.store.db")
    cached_predictions = {
        "probs": np.array([[0.8, 0.2], [0.3, 0.7]]),
        "class_names": ["a", "b"],
        "active_model_mode": "tiny",
    }

    class FakeDB:
        def __init__(self, _path):
            self.path = _path

        def get_most_recent_prediction_cache(self):
            return cached_predictions

        def get_most_recent_model_cache(self):
            return None

    window = MainWindow()
    window.db_path = tmp_path / "classkit.db"
    window._ask_yes_no = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("autoload should not prompt")
    )

    applied = {}

    def fake_apply(payload, db):
        applied["payload"] = payload
        applied["db"] = db

    monkeypatch.setattr(db_module, "ClassKitDB", FakeDB)
    monkeypatch.setattr(window, "_apply_cached_predictions", fake_apply)

    window._autoload_model_from_db()

    assert applied["payload"] is cached_predictions
    assert isinstance(applied["db"], FakeDB)


def test_autoload_model_from_db_prefers_newer_model_over_stale_predictions(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_module = pytest.importorskip("hydra_suite.classkit.core.store.db")
    cached_predictions = {
        "probs": np.array([[0.8, 0.2], [0.3, 0.7]]),
        "class_names": ["a", "b"],
        "active_model_mode": "tiny",
        "timestamp": "2026-04-08 09:00:00",
    }
    recent_model = {
        "id": 42,
        "mode": "tiny",
        "artifact_paths": [str(tmp_path / "latest_model.pt")],
        "timestamp": "2026-04-08 10:00:00",
    }

    class FakeDB:
        def __init__(self, _path):
            self.path = _path

        def get_most_recent_prediction_cache(self):
            return cached_predictions

        def get_most_recent_model_cache(self):
            return recent_model

    window = MainWindow()
    window.db_path = tmp_path / "classkit.db"

    scheduled = {}
    monkeypatch.setattr(db_module, "ClassKitDB", FakeDB)
    monkeypatch.setattr(
        "hydra_suite.classkit.gui.main_window.QTimer.singleShot",
        lambda _ms, callback: callback(),
    )
    monkeypatch.setattr(
        window,
        "_apply_cached_predictions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stale cached predictions should not be used")
        ),
    )

    def fake_load(entry, on_success=None):
        scheduled["entry"] = entry
        scheduled["has_callback"] = callable(on_success)

    monkeypatch.setattr(window, "_load_model_from_cache_entry", fake_load)

    window._autoload_model_from_db()

    assert scheduled["entry"] is recent_model
    assert scheduled["has_callback"] is True


def test_autoload_yolo_classifier_runs_inference_without_prompt(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models_dir = classkit_model_dir(tmp_path)
    models_dir.mkdir(parents=True, exist_ok=True)
    yolo_path = models_dir / "yolo_classifier_latest.pt"
    yolo_path.write_text("weights", encoding="utf-8")

    class FakeDB:
        def get_most_recent_prediction_cache(self):
            return None

    window = MainWindow()
    window.project_path = tmp_path
    window._ask_yes_no = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("autoload should not prompt")
    )

    launched = {}

    monkeypatch.setattr(
        "hydra_suite.classkit.gui.main_window.QTimer.singleShot",
        lambda _ms, callback: callback(),
    )
    window._last_training_settings = {"monochrome": True}
    monkeypatch.setattr(
        window,
        "_run_yolo_inference",
        lambda path, on_success=None, force_monochrome=False: launched.update(
            {"path": path, "force_monochrome": force_monochrome}
        ),
    )

    scheduled = window._autoload_yolo_classifier(FakeDB())

    assert scheduled is True
    assert window._yolo_model_path == yolo_path
    assert window._active_model_mode == "yolo"
    assert launched["path"] == yolo_path
    assert launched["force_monochrome"] is True


def test_load_model_from_cache_entry_dispatches_flat_custom_classifier(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    artifact = tmp_path / "custom_model.pth"
    artifact.write_bytes(b"weights")
    entry = {
        "mode": "flat_custom",
        "artifact_paths": [str(artifact)],
        "class_names": ["left", "right"],
        "meta": {},
    }

    window = MainWindow()
    called = {}

    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: {
            "arch": "timm/convnext_tiny.fb_in1k",
            "class_names": ["left", "right"],
            "input_size": (224, 224),
            "monochrome": True,
        },
    )
    monkeypatch.setattr(
        window,
        "_run_tiny_inference",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(
                "flat custom checkpoints should not route through tiny inference"
            )
        ),
    )
    monkeypatch.setattr(
        window,
        "_run_torchvision_inference",
        lambda model_path, class_names, input_size=224, on_success=None, force_monochrome=False: called.update(
            {
                "path": model_path,
                "class_names": class_names,
                "input_size": input_size,
                "force_monochrome": force_monochrome,
                "has_callback": callable(on_success),
            }
        ),
    )

    window._load_model_from_cache_entry(entry)

    assert called == {
        "path": artifact,
        "class_names": ["left", "right"],
        "input_size": 224,
        "force_monochrome": True,
        "has_callback": True,
    }
    assert window._active_model_mode == "custom_cnn"


def test_load_model_from_cache_entry_dispatches_multihead_custom_classifier(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "factor_a.pth"
    second = tmp_path / "factor_b.pth"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    entry = {
        "mode": "multihead_custom",
        "artifact_paths": [str(first), str(second)],
        "class_names": ["left", "right", "up", "down"],
        "meta": {},
    }

    window = MainWindow()
    called = {}

    monkeypatch.setattr(
        window,
        "_run_multihead_custom_inference",
        lambda model_paths, on_success=None: called.update(
            {
                "paths": model_paths,
                "has_callback": callable(on_success),
            }
        ),
    )
    monkeypatch.setattr(
        window,
        "_run_tiny_inference",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(
                "multi-head custom checkpoints should not route through tiny inference"
            )
        ),
    )

    window._load_model_from_cache_entry(entry)

    assert called == {
        "paths": [first, second],
        "has_callback": True,
    }
    assert window._active_model_mode == "custom_cnn_multihead"


def test_load_checkpoint_from_path_uses_cached_multihead_entry(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = tmp_path / "factor_a.pth"
    sibling = tmp_path / "factor_b.pth"
    selected.write_bytes(b"a")
    sibling.write_bytes(b"b")
    entry = {
        "mode": "multihead_custom",
        "artifact_paths": [str(selected), str(sibling)],
        "class_names": ["left", "right"],
        "meta": {},
    }

    window = MainWindow()
    called = {}

    monkeypatch.setattr(window, "_cached_model_cache_entry", lambda _path: entry)
    monkeypatch.setattr(
        window,
        "_load_model_from_cache_entry",
        lambda cache_entry, on_success=None: called.update(
            {
                "entry": cache_entry,
                "has_callback": callable(on_success),
            }
        ),
    )

    window._load_checkpoint_from_path(selected, show_message_box=False)

    assert called == {
        "entry": entry,
        "has_callback": True,
    }


def test_load_checkpoint_from_path_discovers_external_multihead_manifest(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_module = pytest.importorskip("hydra_suite.classkit.model_bundle")

    selected = tmp_path / "factor_a.pth"
    sibling = tmp_path / "factor_b.pth"
    selected.write_bytes(b"a")
    sibling.write_bytes(b"b")
    bundle_module.write_model_bundle_manifest(
        tmp_path / "external_multihead.bundle.json",
        mode="multihead_custom",
        artifact_paths=[selected, sibling],
        class_names=["left", "right"],
    )

    window = MainWindow()
    called = {}

    monkeypatch.setattr(window, "_cached_model_cache_entry", lambda _path: None)
    monkeypatch.setattr(
        window,
        "_load_model_from_cache_entry",
        lambda entry, on_success=None: called.update(
            {
                "entry": entry,
                "has_callback": callable(on_success),
            }
        ),
    )

    window._load_checkpoint_from_path(selected, show_message_box=False)

    assert called["entry"]["mode"] == "multihead_custom"
    assert called["entry"]["artifact_paths"] == [
        str(selected.resolve()),
        str(sibling.resolve()),
    ]
    assert called["entry"]["class_names"] == ["left", "right"]
    assert called["has_callback"] is True


def test_load_checkpoint_from_path_discovers_external_multihead_siblings(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_module = pytest.importorskip("hydra_suite.classkit.model_bundle")

    selected = tmp_path / "classkit_multihead_custom_20260417_factor_a.pth"
    sibling = tmp_path / "classkit_multihead_custom_20260417_factor_b.pth"
    selected.write_bytes(b"a")
    sibling.write_bytes(b"b")

    window = MainWindow()
    called = {}

    monkeypatch.setattr(window, "_cached_model_cache_entry", lambda _path: None)
    monkeypatch.setattr(
        bundle_module, "_looks_like_classkit_checkpoint", lambda _path: True
    )
    monkeypatch.setattr(
        window,
        "_load_model_from_cache_entry",
        lambda entry, on_success=None: called.update(
            {
                "entry": entry,
                "has_callback": callable(on_success),
            }
        ),
    )

    window._load_checkpoint_from_path(selected, show_message_box=False)

    assert called["entry"]["mode"] == "multihead_custom"
    assert called["entry"]["artifact_paths"] == [
        str(selected.resolve()),
        str(sibling.resolve()),
    ]
    assert called["has_callback"] is True


def test_post_training_inference_dispatches_multihead_custom_classifier(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "factor_a.pth"
    second = tmp_path / "factor_b.pth"
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    class Dialog:
        def __init__(self) -> None:
            self.logs = []

        def append_log(self, message: str) -> None:
            self.logs.append(message)

    window = MainWindow()
    dialog = Dialog()
    called = {}

    monkeypatch.setattr(
        window,
        "_run_multihead_custom_inference",
        lambda model_paths, on_success=None: called.update(
            {
                "paths": model_paths,
                "has_callback": callable(on_success),
            }
        ),
    )

    window._run_post_training_inference(
        [
            {"artifact_path": str(first)},
            {"artifact_path": str(second)},
        ],
        is_yolo=False,
        multi_head=True,
        labels_str=["left", "right"],
        dialog=dialog,
        on_done=lambda *_args, **_kwargs: None,
    )

    assert called == {
        "paths": [first, second],
        "has_callback": True,
    }
    assert window._active_model_mode == "custom_cnn_multihead"
    assert dialog.logs == ["Running multi-head Custom CNN inference (2 models)..."]


def test_apply_cached_predictions_switches_to_prediction_view_when_no_batch(
    qapp,
) -> None:
    class FakeDB:
        def get_most_recent_umap_cache(self, kind="model"):
            return None

    window = MainWindow()
    window.image_paths = [Path("/tmp/a.png"), Path("/tmp/b.png")]
    window.umap_coords = np.array([[0.0, 0.0], [1.0, 1.0]])

    window._apply_cached_predictions(
        {
            "probs": np.array([[0.8, 0.2], [0.3, 0.7]]),
            "class_names": ["a", "b"],
            "active_model_mode": "tiny",
        },
        FakeDB(),
    )

    assert window.explorer_mode == "predictions"


def test_apply_cached_predictions_restores_metrics_without_switching_tabs(qapp) -> None:
    class FakeDB:
        def get_most_recent_umap_cache(self, kind="model"):
            return None

        def get_most_recent_model_cache(self):
            return {"best_val_acc": 0.875}

    window = MainWindow()
    window.image_paths = [Path("/tmp/a.png"), Path("/tmp/b.png")]
    window.image_labels = ["class_1", "class_2"]
    window.classes = ["class_1", "class_2"]
    window.umap_coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    current_tab = window.tabs.currentWidget()

    window._apply_cached_predictions(
        {
            "probs": np.array([[0.9, 0.1], [0.2, 0.8]]),
            "class_names": ["class_1", "class_2"],
            "active_model_mode": "tiny",
        },
        FakeDB(),
    )

    assert "Classification Metrics" in window.metrics_view.toPlainText()
    assert "0.875" in window.metrics_validation_label.text()
    assert window.tabs.currentWidget() is current_tab


def test_metrics_display_includes_heldout_validation_summary(qapp) -> None:
    metrics_module = pytest.importorskip("hydra_suite.classkit.core.train.metrics")

    window = MainWindow()
    metrics = metrics_module.compute_metrics(
        np.array([0, 1]),
        np.array([0, 1]),
        class_names=["class_1", "class_2"],
    )
    window._set_heldout_validation_summary(
        window._validation_summary_from_results([{"best_val_acc": 0.875}])
    )

    window._update_metrics_display(metrics, activate_metrics_tab=False)

    assert "0.875" in window.metrics_validation_label.text()
    assert "0.875" in window.metrics_view.toPlainText()


def test_validation_summary_uses_multihead_factor_names(qapp) -> None:
    window = MainWindow()

    summary = window._validation_summary_from_results(
        [{"best_val_acc": 0.91}, {"best_val_acc": 0.84}],
        factor_names=["color", "side"],
    )

    assert summary is not None
    assert "color=0.910" in summary["text"]
    assert "side=0.840" in summary["text"]
    assert "mean=0.875" in summary["text"]


def test_metrics_display_includes_multihead_exact_match_and_factor_accuracy(
    qapp,
) -> None:
    window = MainWindow()
    window.image_paths = [Path("/tmp/a.png"), Path("/tmp/b.png"), Path("/tmp/c.png")]
    window.image_labels = ["blue|left", "red|right", "blue|right"]
    window.classes = ["blue", "red"]
    window._set_model_prediction_state(
        np.array(
            [
                [0.90, 0.10, 0.85, 0.15],
                [0.20, 0.80, 0.10, 0.90],
                [0.85, 0.15, 0.70, 0.30],
            ],
            dtype=np.float32,
        ),
        ["blue", "red", "left", "right"],
        active_model_mode="custom_cnn_multihead",
        prediction_heads=[
            {"factor": "color", "class_names": ["blue", "red"]},
            {"factor": "side", "class_names": ["left", "right"]},
        ],
    )

    window._evaluate_model_on_labeled(activate_metrics_tab=False)

    metrics_text = window.metrics_view.toPlainText()
    assert "Multi-head summary" in metrics_text
    assert "Exact-match accuracy: 0.667" in metrics_text
    assert "color: acc=1.000" in metrics_text
    assert "side: acc=0.667" in metrics_text


def test_metrics_display_includes_train_val_and_test_split_reports(
    qapp, tmp_path: Path
) -> None:
    metrics_module = pytest.importorskip("hydra_suite.classkit.core.train.metrics")

    window = MainWindow()
    image_paths = [tmp_path / f"img_{idx}.png" for idx in range(6)]
    window.image_paths = [str(path) for path in image_paths]
    window.image_labels = [
        "class_1",
        "class_1",
        "class_2",
        "class_2",
        "class_1",
        "class_2",
    ]
    window.classes = ["class_1", "class_2"]
    window._model_class_names = ["class_1", "class_2"]
    window._model_probs = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.3, 0.7],
        ]
    )
    window._set_active_evaluation_selection(
        [str(image_paths[4]), str(image_paths[5])],
        "test",
        split_paths={
            "train": [str(image_paths[0]), str(image_paths[2])],
            "val": [str(image_paths[1]), str(image_paths[3])],
            "test": [str(image_paths[4]), str(image_paths[5])],
        },
    )
    metrics = metrics_module.compute_metrics(
        np.array([0, 1]),
        np.array([0, 1]),
        class_names=["class_1", "class_2"],
    )

    window._update_metrics_display(
        metrics,
        evaluation_scope="Test split",
        activate_metrics_tab=False,
    )

    report = window.metrics_view.toPlainText()
    assert "Split Summary" in report
    assert "Train    acc=" in report
    assert "Val      acc=" in report
    assert "Test     acc=" in report
    assert report.count("Classification Metrics") >= 4


def test_metrics_display_avoids_divide_warning_for_empty_confusion_rows(qapp) -> None:
    metrics_module = pytest.importorskip("hydra_suite.classkit.core.train.metrics")

    window = MainWindow()
    metrics = metrics_module.EvalMetrics(
        accuracy=1.0,
        macro_precision=0.5,
        macro_recall=0.5,
        macro_f1=0.5,
        weighted_f1=1.0,
        per_class=[
            metrics_module.ClassMetrics(
                class_id=0,
                class_name="class_1",
                precision=1.0,
                recall=1.0,
                f1=1.0,
                support=2,
            ),
            metrics_module.ClassMetrics(
                class_id=1,
                class_name="class_2",
                precision=0.0,
                recall=0.0,
                f1=0.0,
                support=0,
            ),
        ],
        confusion_matrix=np.array([[2, 0], [0, 0]], dtype=int),
        num_samples=2,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        window._update_metrics_display(metrics, activate_metrics_tab=False)

    assert not any(
        "invalid value encountered in divide" in str(warning.message)
        for warning in caught
    )


def test_load_project_data_clears_stale_model_state_before_restore(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_module = pytest.importorskip("hydra_suite.classkit.core.store.db")

    class FakeDB:
        def __init__(self, _path):
            self.path = _path

        def migrate_paths_to_resolved(self):
            return 0

        def get_all_image_paths(self):
            return [str(tmp_path / "a.png"), str(tmp_path / "b.png")]

        def get_all_labels(self):
            return ["class_1", "class_2"]

        def get_most_recent_embeddings(self):
            return None

        def get_most_recent_cluster_cache(self, embedding_cache_id=None):
            return None

        def get_most_recent_umap_cache(self, embedding_cache_id=None, kind="embedding"):
            return None

        def get_most_recent_candidate_cache(self):
            return None

        def get_most_recent_prediction_cache(self):
            return {
                "probs": np.array([[0.9, 0.1], [0.2, 0.8]]),
                "class_names": ["class_1", "class_2"],
                "active_model_mode": "tiny",
            }

        def get_most_recent_model_cache(self):
            return None

    window = MainWindow()
    window.project_path = tmp_path
    window.db_path = tmp_path / "classkit.db"
    window._model_probs = np.array([[0.1, 0.9], [0.8, 0.2]])
    window.metrics_view.setPlainText("stale metrics")

    monkeypatch.setattr(db_module, "ClassKitDB", FakeDB)
    monkeypatch.setattr(window, "_load_project_config", lambda: {})

    window.load_project_data()

    assert "Classification Metrics" in window.metrics_view.toPlainText()
    assert np.allclose(window._model_probs, np.array([[0.9, 0.1], [0.2, 0.8]]))


def test_activate_saved_labels_view_switches_to_labeling_when_labels_exist(
    qapp,
) -> None:
    window = MainWindow()
    window.image_paths = [Path("/tmp/a.png"), Path("/tmp/b.png")]
    window.image_labels = ["alpha", None]
    window.umap_coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    window.explorer_mode = "explore"

    window._activate_saved_labels_view_if_available()

    assert window.explorer_mode == "labeling"


def test_activate_saved_labels_view_does_not_override_predictions_or_batches(
    qapp,
) -> None:
    window = MainWindow()
    window.image_paths = [Path("/tmp/a.png"), Path("/tmp/b.png")]
    window.image_labels = ["alpha", None]
    window.umap_coords = np.array([[0.0, 0.0], [1.0, 1.0]])

    window._model_probs = np.array([[0.8, 0.2], [0.3, 0.7]])
    window.explorer_mode = "predictions"
    window._activate_saved_labels_view_if_available()
    assert window.explorer_mode == "predictions"

    window._model_probs = None
    window.candidate_indices = [1]
    window.explorer_mode = "explore"
    window._activate_saved_labels_view_if_available()
    assert window.explorer_mode == "explore"


def test_open_recent_project_flushes_pending_label_updates_before_switch(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "classkit.db").write_text("db", encoding="utf-8")

    window = MainWindow()
    calls = []

    monkeypatch.setattr(
        window,
        "_flush_pending_label_updates",
        lambda force=False: calls.append(("flush", force)),
    )
    monkeypatch.setattr(
        window,
        "load_project_data",
        lambda: calls.append(("load", str(window.project_path))),
    )
    monkeypatch.setattr(
        window,
        "update_context_panel",
        lambda: calls.append(("context", None)),
    )

    window._open_recent_project(str(project_dir))

    assert calls[0] == ("flush", True)
    assert calls[1] == ("load", str(project_dir))
    assert calls[1] == ("load", str(project_dir))
