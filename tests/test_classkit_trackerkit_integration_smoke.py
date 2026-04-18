from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDialog

from hydra_suite.trackerkit.gui.main_window import MainWindow
from hydra_suite.training.contracts import (
    SourceDataset,
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)
from hydra_suite.training.service import TrainingOrchestrator


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_main_window(monkeypatch: pytest.MonkeyPatch) -> MainWindow:
    monkeypatch.setattr(MainWindow, "_save_advanced_config", lambda self: None)
    monkeypatch.setattr(MainWindow, "_load_advanced_config", lambda self: {})
    return MainWindow()


def test_classkit_multiartifact_publish_is_discoverable_and_selectable_in_trackerkit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tiny_flat_subset: Path,
) -> None:
    import hydra_suite.trackerkit.gui.dialogs.cnn_identity_import_dialog as dialog_module
    import hydra_suite.training.model_publish as model_publish
    import hydra_suite.training.service as service_module

    data_dir = tmp_path / "hydra-data"
    monkeypatch.setenv("HYDRA_DATA_DIR", str(data_dir))

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset.yaml").write_text(
        "train: images/train\nval: images/val\n",
        encoding="utf-8",
    )

    artifact_a = tmp_path / "artifacts" / "color.pth"
    artifact_b = tmp_path / "artifacts" / "shape.pth"
    artifact_a.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(tiny_flat_subset), str(artifact_a))
    shutil.copy2(str(tiny_flat_subset), str(artifact_b))

    def _fake_run_training(spec, run_dir, **kwargs):
        return {
            "success": True,
            "command": ["train"],
            "artifact_paths": [str(artifact_a), str(artifact_b)],
        }

    monkeypatch.setattr(service_module, "run_training", _fake_run_training)

    spec = TrainingRunSpec(
        role=TrainingRole.CLASSIFY_MULTIHEAD_TINY,
        source_datasets=[SourceDataset(path=str(dataset_dir))],
        derived_dataset_dir=str(dataset_dir),
        base_model="tinyclassifier",
        hyperparams=TrainingHyperParams(),
        device="cpu",
    )
    result = TrainingOrchestrator(tmp_path / "workspace").run_role_training(
        spec,
        publish_metadata={
            "size": "tiny",
            "species": "ant",
            "model_info": "smoke_bundle",
            "scheme_name": "smoke_scheme",
            "factor_names": ["color", "shape"],
        },
    )

    published_path = Path(result["published_model_path"])
    assert published_path.exists()
    assert published_path.name.endswith(".multihead.json")

    models_root = model_publish.get_models_root()
    rel_path = published_path.relative_to(models_root).as_posix()
    assert rel_path.startswith("tiny-classify/multihead/smoke_scheme/")

    class _FakeDialog:
        def __init__(self, summary, parent=None):
            assert summary["is_multihead"] is True
            assert summary["factor_names"] == ["color", "shape"]

        def exec(self) -> int:
            return QDialog.Accepted

        def species(self) -> str:
            return "ant"

        def classification_label(self) -> str:
            return "smoke_tags"

        def scoring_mode(self) -> str:
            return "per_head_average"

    monkeypatch.setattr(dialog_module, "CNNIdentityImportDialog", _FakeDialog)

    window = _make_main_window(monkeypatch)
    try:
        window._identity_panel._refresh_cnn_identity_model_combo()
        combo = window._identity_panel.combo_cnn_identity_model
        combo_index = combo.findData(rel_path)
        assert combo_index >= 0
        assert combo.itemText(combo_index).startswith("[classkit]")

        combo.blockSignals(True)
        combo.setCurrentIndex(combo_index)
        combo.blockSignals(False)
        window._identity_panel._on_cnn_identity_model_selected(combo_index)

        entries = dict(model_publish.iter_registry_entries())
        assert rel_path in entries
        entry = entries[rel_path]
        assert entry["usage_role"] == "cnn_identity"
        assert entry["classification_label"] == "smoke_tags"
        assert entry["scoring_mode"] == "per_head_average"
        assert entry["factor_names"] == ["color", "shape"]

        assert combo.findData(rel_path) >= 0
        assert combo.currentData() == rel_path
        assert window._identity_panel.lbl_cnn_label.text() == "smoke_tags"
        assert window._identity_panel.lbl_cnn_num_classes.text() == "4"
    finally:
        window.close()
