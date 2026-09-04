"""Tests for DetectKit TrainingDialog — full feature set."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture(autouse=True)
def _hermetic_ui_settings(tmp_path_factory, monkeypatch):
    """Isolate DetectKit persistent UI settings to a clean temp dir.

    TrainingDialog._apply_persistent_state() reads ui_settings.json (under the
    data dir) and overrides per-project values. Point HYDRA_DATA_DIR/CONFIG_DIR
    at a fresh temp dir so the developer's real ui_settings.json can't clobber
    the project values these tests assert on (get_ui_settings_path() reads the
    env var at call time).
    """
    home = tmp_path_factory.mktemp("hydra_home")
    monkeypatch.setenv("HYDRA_DATA_DIR", str(home / "data"))
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(home / "config"))
    yield


def _make_proj(tmp_path):
    from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource

    proj = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    proj.sources = [OBBSource(path=str(tmp_path / "ds1"), name="ds1")]
    return proj


def _write_detectkit_source_dataset(root: Path) -> Path:
    images_dir = root / "images" / "train"
    labels_dir = root / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_path = images_dir / "sample.png"
    image = QImage(320, 240, QImage.Format.Format_RGB32)
    image.fill(0)
    assert image.save(str(image_path))
    (labels_dir / "sample.txt").write_text(
        "0 0.10 0.10 0.40 0.10 0.40 0.40 0.10 0.40\n",
        encoding="utf-8",
    )
    (root / "dataset.yaml").write_text(
        "train: images/train\nval: images/train\nnames:\n  0: ant\n",
        encoding="utf-8",
    )
    return root


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


def test_training_dialog_is_base_dialog(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog
    from hydra_suite.widgets.dialogs import BaseDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    assert isinstance(dlg, BaseDialog)


def test_training_dialog_has_close_button(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    close_btn = dlg._buttons.button(QDialogButtonBox.StandardButton.Close)
    assert close_btn is not None


def test_training_dialog_has_overview_tabs(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "training_tabs")
    # Overview, Advanced, and the SAM3 finetuning tab.
    assert dlg.training_tabs.count() == 3


def test_training_dialog_uses_advanced_tab_label(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    assert dlg.training_tabs.tabText(1) == "Advanced"


def test_training_dialog_uses_compact_grid_layouts(qapp, tmp_path):
    """The main controls stay dense instead of expanding into long forms."""
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    groups = {group.title(): group for group in dlg.findChildren(QGroupBox)}
    assert isinstance(groups["Training Selection"].layout(), QGridLayout)
    assert isinstance(groups["Dataset And Runtime"].layout(), QGridLayout)
    assert isinstance(dlg.slice_group.layout(), QHBoxLayout)


def test_training_dialog_interactive_controls_have_tooltips(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dialog = TrainingDialog(_make_proj(tmp_path))
    interactive = (
        QAbstractButton,
        QAbstractSpinBox,
        QComboBox,
        QLineEdit,
    )
    controls = [
        control
        for control_type in interactive
        for control in dialog.findChildren(control_type)
        if control.isEnabled()
    ]
    assert controls
    assert all(control.toolTip() for control in controls)


def test_training_dialog_summary_reflects_current_plan(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.class_names = ["worker", "queen"]
    proj.species = "ant"
    proj.model_tag = "v2"
    dlg = TrainingDialog(proj)

    summary = dlg.plan_summary.text()
    assert "Plan:" in summary
    assert "Stages:" in summary
    assert "Classes:" in summary
    assert "worker" in summary


def test_training_dialog_has_two_simple_plan_selectors(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "mode_combo")
    assert hasattr(dlg, "task_combo")
    # "Direct", "Sequential", and "Semantic" (SAM3 finetuning).
    assert dlg.mode_combo.count() == 3
    assert dlg.task_combo.count() == 3


def test_training_dialog_defaults_to_one_direct_obb_plan(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    assert dlg.mode_combo.currentData() == "direct"
    assert dlg.task_combo.currentData() == "obb"
    assert dlg.chk_role_obb_direct.isChecked() is True
    assert dlg.chk_role_seq_detect.isChecked() is False
    assert dlg.chk_role_seq_crop_obb.isChecked() is False


def test_training_dialog_sequential_selection_updates_roles(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    dlg.mode_combo.setCurrentIndex(dlg.mode_combo.findData("sequential"))
    dlg.task_combo.setCurrentIndex(dlg.task_combo.findData("obb"))

    assert dlg.chk_role_obb_direct.isChecked() is False
    assert dlg.chk_role_seq_detect.isChecked() is True
    assert dlg.chk_role_seq_crop_obb.isChecked() is True


def test_training_dialog_selection_maps_each_task_to_one_valid_plan(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.sources[0].level = "polygon"
    dlg = TrainingDialog(proj)
    dlg.mode_combo.setCurrentIndex(dlg.mode_combo.findData("direct"))
    dlg.task_combo.setCurrentIndex(dlg.task_combo.findData("segment"))

    assert dlg.chk_role_segment_direct.isChecked() is True
    assert dlg.chk_role_obb_direct.isChecked() is False
    assert dlg.chk_role_seq_detect.isChecked() is False


def test_training_dialog_start_always_enabled(qapp, tmp_path):
    """Start button is enabled by default; sources are validated at click time."""
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    assert dlg.btn_start.isEnabled()


def test_training_worker_is_base_worker(qapp):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import _TrainingWorker
    from hydra_suite.widgets.workers import BaseWorker

    assert issubclass(_TrainingWorker, BaseWorker)


def test_training_worker_retains_owned_sidecar_until_recovery(qapp, monkeypatch):
    from types import SimpleNamespace

    from hydra_suite.detectkit.gui.dialogs import training_dialog as td
    from hydra_suite.runtime.process_supervisor import WorkloadStillOwnedError
    from hydra_suite.training.contracts import TrainingRole

    class _Sidecar:
        def __init__(self):
            self.fail_cleanup = True
            self.generic_failure = False

        def cancel(self):
            if self.generic_failure:
                raise OSError("signal transport failed")
            if self.fail_cleanup:
                raise WorkloadStillOwnedError("still owned", self)

    sidecar = _Sidecar()
    owned = WorkloadStillOwnedError("ownership retained", sidecar)
    owned.run_id = "run-owned"
    recovered_artifacts = []
    owned.recovery_cleanup = lambda: recovered_artifacts.append(True)
    calls = []
    finalized = []
    monkeypatch.setattr(
        td,
        "finalize_run_record",
        lambda run_id, **kwargs: finalized.append((run_id, kwargs)),
    )

    class _Orchestrator:
        def run_role_training(self, spec, **_kwargs):
            calls.append(spec.role)
            raise owned

    entries = [
        SimpleNamespace(
            role=TrainingRole.SEMANTIC_SAM3,
            spec=SimpleNamespace(role=TrainingRole.SEMANTIC_SAM3),
            publish_metadata={},
        ),
        SimpleNamespace(
            role=TrainingRole.DETECT_DIRECT,
            spec=SimpleNamespace(role=TrainingRole.DETECT_DIRECT),
            publish_metadata={},
        ),
    ]
    worker = td._TrainingWorker(_Orchestrator(), entries)
    completed = []
    worker.done_signal.connect(completed.append)

    worker.run()

    assert worker.failure_exception is owned
    assert worker.containment_recovery_required
    assert completed == []
    assert calls == [TrainingRole.SEMANTIC_SAM3]
    assert not worker.retry_containment_cleanup()
    assert worker.containment_recovery_required
    assert worker.failure_exception.sidecar is sidecar
    assert recovered_artifacts == []

    sidecar.fail_cleanup = False
    sidecar.generic_failure = True
    retained = worker.failure_exception
    assert not worker.retry_containment_cleanup()
    assert worker.failure_exception is retained
    assert worker.failure_exception.recovery_error == "signal transport failed"
    assert recovered_artifacts == []

    sidecar.generic_failure = False
    assert worker.retry_containment_cleanup()
    assert worker.failure_exception is None
    assert recovered_artifacts == [True]
    assert finalized[0][0] == "run-owned"
    assert finalized[0][1]["status"] == "failed"
    assert finalized[0][1]["failure_details"]["containment"]["ownership"] == (
        "recovered"
    )


def test_training_worker_reports_registry_failure_after_safe_cleanup(qapp, monkeypatch):
    from hydra_suite.detectkit.gui.dialogs import training_dialog as td
    from hydra_suite.runtime.process_supervisor import WorkloadStillOwnedError

    sidecar = type("Sidecar", (), {"cancel": lambda self: None})()
    owned = WorkloadStillOwnedError("ownership retained", sidecar)
    owned.run_id = "run-owned"
    owned.recovery_cleanup = lambda: (_ for _ in ()).throw(
        OSError("staging cleanup failed")
    )
    worker = td._TrainingWorker(object(), [])
    worker.failure_exception = owned
    monkeypatch.setattr(
        td,
        "finalize_run_record",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("registry offline")),
    )

    assert worker.retry_containment_cleanup()
    assert worker.failure_exception is None
    assert worker.recovery_registry_error == "registry offline"
    assert worker.recovery_cleanup_error == "staging cleanup failed"


def test_resume_builds_shared_role_entry(qapp, tmp_path, monkeypatch):
    from hydra_suite.detectkit.gui.dialogs import training_dialog as td
    from hydra_suite.detectkit.jobs.training import RoleTrainingEntry

    run_dir = tmp_path / "runs" / "previous"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "last.pt").write_bytes(b"checkpoint")

    dialog = td.TrainingDialog(_make_proj(tmp_path))
    dialog._last_training_results = [
        {
            "role": "obb_direct",
            "_run_dir": str(run_dir),
            "derived_dataset_dir": str(tmp_path / "derived"),
        }
    ]
    monkeypatch.setattr(dialog, "_get_orchestrator", lambda: object())
    captured = {}

    class _Signal:
        def connect(self, *_args, **_kwargs):
            pass

    class _Worker:
        def __init__(self, _orchestrator, entries):
            captured["entries"] = entries
            self.log_signal = _Signal()
            self.role_started = _Signal()
            self.role_finished = _Signal()
            self.progress_signal = _Signal()
            self.done_signal = _Signal()
            self.finished = _Signal()

        def start(self):
            pass

    monkeypatch.setattr(td, "_TrainingWorker", _Worker)

    dialog._resume_training()

    entry = captured["entries"][0]
    assert isinstance(entry, RoleTrainingEntry)
    assert entry.spec.derived_dataset_dir == str(tmp_path / "derived")


def test_dataset_preparation_worker_is_base_worker(qapp):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import (
        _DatasetPreparationWorker,
    )
    from hydra_suite.widgets.workers import BaseWorker

    assert issubclass(_DatasetPreparationWorker, BaseWorker)


def test_dataset_preparation_runs_builders_off_gui_thread(qapp):
    from types import SimpleNamespace

    from PySide6.QtCore import QThread

    from hydra_suite.detectkit.gui.dialogs.training_dialog import (
        _DatasetPreparationRequest,
        _DatasetPreparationWorker,
    )
    from hydra_suite.detectkit.gui.models import SliceTrainingSettings
    from hydra_suite.training.contracts import SourceDataset, SplitConfig, TrainingRole

    called_from = []

    class _Orchestrator:
        def build_merged_obb_dataset(self, *_args, **_kwargs):
            called_from.append(QThread.currentThread())
            return SimpleNamespace(
                dataset_dir="/tmp/merged",
                stats={"source_items": {"source": 1}},
            )

        def build_role_dataset(self, *_args, **_kwargs):
            called_from.append(QThread.currentThread())
            return SimpleNamespace(dataset_dir="/tmp/role")

    request = _DatasetPreparationRequest(
        sources=(SourceDataset(path="/tmp/source"),),
        roles=(TrainingRole.OBB_DIRECT,),
        class_names=("ant",),
        split=SplitConfig(),
        seed=7,
        dedup=True,
        crop_pad_ratio=0.15,
        min_crop_size_px=128,
        enforce_square=True,
        imgsz_by_role=((TrainingRole.OBB_DIRECT.value, 640),),
        slice_settings=SliceTrainingSettings(enabled=False),
    )
    errors = []
    worker = _DatasetPreparationWorker(_Orchestrator(), request)
    worker.error.connect(errors.append)

    worker.start()
    assert worker.wait(5000), "dataset preparation worker did not finish"
    qapp.processEvents()

    assert errors == []
    assert len(called_from) == 2
    assert all(thread is not qapp.thread() for thread in called_from)


def test_dataset_preparation_preserves_merge_reuse_and_slice_routing(qapp):
    from types import SimpleNamespace

    from hydra_suite.detectkit.gui.dialogs.training_dialog import (
        _DatasetPreparationRequest,
        _prepare_role_datasets,
    )
    from hydra_suite.detectkit.gui.models import SliceTrainingSettings
    from hydra_suite.training.contracts import SourceDataset, SplitConfig, TrainingRole

    calls = {"merge": [], "slice": [], "role": []}

    class _Orchestrator:
        def build_merged_obb_dataset(self, *_args, **kwargs):
            calls["merge"].append(kwargs)
            return SimpleNamespace(
                dataset_dir="/tmp/merged",
                stats={"source_items": {"source": 1}},
            )

        def build_sliced_obb_dataset(self, source_dir, **kwargs):
            calls["slice"].append((source_dir, kwargs))
            return SimpleNamespace(
                dataset_dir="/tmp/sliced",
                stats={"measured_reference_body_px": 42.5},
            )

        def build_role_dataset(self, role, source_dir, **kwargs):
            calls["role"].append((role, source_dir, kwargs))
            return SimpleNamespace(dataset_dir=f"/tmp/{role.value}")

    request = _DatasetPreparationRequest(
        sources=(SourceDataset(path="/tmp/source", level="aabb"),),
        roles=(TrainingRole.DETECT_DIRECT, TrainingRole.SEQ_DETECT),
        class_names=("ant",),
        split=SplitConfig(),
        seed=7,
        dedup=True,
        crop_pad_ratio=0.15,
        min_crop_size_px=128,
        enforce_square=True,
        imgsz_by_role=((TrainingRole.DETECT_DIRECT.value, 768),),
        slice_settings=SliceTrainingSettings(enabled=True),
    )

    result = _prepare_role_datasets(
        _Orchestrator(),
        request,
        log=lambda _message: None,
        status=lambda _message: None,
        should_cancel=lambda: False,
    )

    assert len(calls["merge"]) == 1
    assert len(calls["slice"]) == 1
    assert calls["slice"][0][0] == "/tmp/merged"
    assert calls["slice"][0][1]["params"].imgsz == 768
    assert calls["slice"][0][1]["params"].target_sizes == [240.0, 360.0, 480.0]
    assert calls["slice"][0][1]["params"].reference_body_px == 0.0
    assert [source_dir for _, source_dir, _ in calls["role"]] == [
        "/tmp/sliced",
        "/tmp/merged",
    ]
    assert result.role_dataset_dirs == {
        TrainingRole.DETECT_DIRECT.value: "/tmp/detect_direct",
        TrainingRole.SEQ_DETECT.value: "/tmp/seq_detect",
    }
    assert result.measured_reference_body_px == 42.5


def test_start_training_launches_dataset_preparation_without_building_inline(
    qapp, tmp_path, monkeypatch
):
    from hydra_suite.detectkit.gui.dialogs import training_dialog as td

    class _Signal:
        def connect(self, *_args, **_kwargs):
            pass

    class _FakePreparationWorker:
        def __init__(self, orchestrator, request):
            self.orchestrator = orchestrator
            self.request = request
            self.log_signal = _Signal()
            self.status = _Signal()
            self.result_ready = _Signal()
            self.error = _Signal()
            self.finished = _Signal()
            self.started = False

        def isRunning(self):
            return self.started

        def start(self):
            self.started = True

        def cancel(self):
            pass

    class _Orchestrator:
        def build_merged_obb_dataset(self, *_args, **_kwargs):
            raise AssertionError("dataset builder ran synchronously")

        def build_role_dataset(self, *_args, **_kwargs):
            raise AssertionError("role builder ran synchronously")

    dlg = td.TrainingDialog(_make_proj(tmp_path))
    orchestrator = _Orchestrator()
    monkeypatch.setattr(dlg, "_get_orchestrator", lambda: orchestrator)
    monkeypatch.setattr(td, "_DatasetPreparationWorker", _FakePreparationWorker)

    dlg._start_training()

    worker = dlg._dataset_worker
    assert isinstance(worker, _FakePreparationWorker)
    assert worker.started is True
    assert worker.orchestrator is orchestrator
    assert dlg._worker is None
    assert dlg._training_running is True
    assert (dlg.progress.minimum(), dlg.progress.maximum()) == (0, 0)

    dlg._dataset_worker = None
    dlg._set_training_running(False)
    dlg.close()


def test_cancel_routes_to_active_dataset_preparation(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    class _PreparationWorker:
        cancelled = False

        def cancel(self):
            self.cancelled = True

    dlg = TrainingDialog(_make_proj(tmp_path))
    worker = _PreparationWorker()
    dlg._dataset_worker = worker

    dlg._cancel_training()

    assert worker.cancelled is True
    assert "safe dataset boundary" in dlg.run_status_label.text()

    dlg._dataset_worker = None
    dlg.close()


def test_cancelled_preparation_never_transitions_to_training(
    qapp, tmp_path, monkeypatch
):
    from hydra_suite.detectkit.gui.dialogs import training_dialog as td
    from hydra_suite.training.contracts import TrainingRole

    class _CancelledWorker:
        def is_cancelled(self):
            return True

    dlg = td.TrainingDialog(_make_proj(tmp_path))
    dlg._dataset_worker = _CancelledWorker()
    dlg._pending_dataset_result = td._DatasetPreparationResult(
        role_dataset_dirs={TrainingRole.OBB_DIRECT.value: "/tmp/role"},
        roles=(TrainingRole.OBB_DIRECT,),
    )
    started = []
    monkeypatch.setattr(dlg, "_start_training_worker", started.append)

    dlg._on_dataset_worker_finished()

    assert started == []
    assert dlg._training_running is False
    assert dlg.progress.format() == "Cancelled"
    dlg.close()


# ---------------------------------------------------------------------------
# Roles group
# ---------------------------------------------------------------------------


def test_training_dialog_plan_roundtrip(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.training_mode = "sequential"
    proj.training_task = "segment"
    proj.sources[0].level = "polygon"
    dlg = TrainingDialog(proj)
    assert dlg.mode_combo.currentData() == "sequential"
    assert dlg.task_combo.currentData() == "segment"
    assert dlg.chk_role_seq_detect.isChecked() is True
    assert dlg.chk_role_seq_crop_segment.isChecked() is True


# ---------------------------------------------------------------------------
# Config group
# ---------------------------------------------------------------------------


def test_training_dialog_has_seed_spinbox(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.seed = 77
    dlg = TrainingDialog(proj)
    assert hasattr(dlg, "spin_seed")
    assert dlg.spin_seed.value() == 77


def test_training_dialog_has_dedup_checkbox(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.dedup = False
    dlg = TrainingDialog(proj)
    assert hasattr(dlg, "chk_dedup")
    assert dlg.chk_dedup.isChecked() is False


def test_training_dialog_has_crop_derivation_widgets(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.crop_pad_ratio = 0.25
    proj.min_crop_size_px = 128
    proj.enforce_square = False
    dlg = TrainingDialog(proj)
    assert hasattr(dlg, "spin_crop_pad")
    assert hasattr(dlg, "spin_crop_min_px")
    assert hasattr(dlg, "chk_crop_square")
    assert abs(dlg.spin_crop_pad.value() - 0.25) < 0.001
    assert dlg.spin_crop_min_px.value() == 128
    assert dlg.chk_crop_square.isChecked() is False


# ---------------------------------------------------------------------------
# Hyperparams group — extended fields
# ---------------------------------------------------------------------------


def test_training_dialog_has_workers_cache_auto_batch(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.workers = 4
    proj.cache = True
    proj.auto_batch = True
    dlg = TrainingDialog(proj)
    assert hasattr(dlg, "spin_workers")
    assert hasattr(dlg, "chk_cache")
    assert hasattr(dlg, "chk_auto_batch")
    assert dlg.spin_workers.value() == 4
    assert dlg.chk_cache.isChecked() is True
    assert dlg.chk_auto_batch.isChecked() is True


def test_training_dialog_has_per_role_imgsz(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.imgsz_obb_direct = 800
    proj.imgsz_seq_detect = 960
    proj.imgsz_seq_crop_obb = 256
    dlg = TrainingDialog(proj)
    assert hasattr(dlg, "spin_imgsz_obb_direct")
    assert hasattr(dlg, "spin_imgsz_seq_detect")
    assert hasattr(dlg, "spin_imgsz_seq_crop_obb")
    assert dlg.spin_imgsz_obb_direct.value() == 800
    assert dlg.spin_imgsz_seq_detect.value() == 960
    assert dlg.spin_imgsz_seq_crop_obb.value() == 256


def test_training_dialog_load_from_project_populates_fields(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.epochs = 50
    proj.split_train = 0.7
    dlg = TrainingDialog(proj)
    assert dlg.spin_epochs.value() == 50


# ---------------------------------------------------------------------------
# Base Models group
# ---------------------------------------------------------------------------


def test_training_dialog_has_per_role_model_combos(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.model_obb_direct = "yolo26m-obb.pt"
    proj.model_seq_detect = "yolo26n.pt"
    proj.model_seq_crop_obb = "yolo26n-obb.pt"
    dlg = TrainingDialog(proj)
    assert hasattr(dlg, "combo_model_obb_direct")
    assert hasattr(dlg, "combo_model_seq_detect")
    assert hasattr(dlg, "combo_model_seq_crop_obb")
    assert dlg.combo_model_obb_direct.currentText() == "yolo26m-obb.pt"
    assert dlg.combo_model_seq_detect.currentText() == "yolo26n.pt"
    assert dlg.combo_model_seq_crop_obb.currentText() == "yolo26n-obb.pt"


def test_training_dialog_recipe_hides_irrelevant_advanced_controls(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    dlg.mode_combo.setCurrentIndex(dlg.mode_combo.findData("direct"))
    dlg.task_combo.setCurrentIndex(dlg.task_combo.findData("obb"))

    assert dlg.spin_imgsz_obb_direct.isHidden() is False
    assert dlg.spin_imgsz_seq_detect.isHidden() is True
    assert dlg.spin_imgsz_seq_crop_obb.isHidden() is True
    assert dlg.combo_model_obb_direct.isHidden() is False
    assert dlg.combo_model_seq_detect.isHidden() is True
    assert dlg.combo_model_seq_crop_obb.isHidden() is True
    assert dlg.crop_settings_widget.isHidden() is True


def test_training_dialog_source_preview_loads_real_source_samples(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog
    from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource

    source_root = _write_detectkit_source_dataset(tmp_path / "preview_source")
    additional_image = source_root / "images" / "train" / "wide.png"
    image = QImage(640, 320, QImage.Format.Format_RGB32)
    image.fill(0)
    assert image.save(str(additional_image))
    (source_root / "labels" / "train" / "wide.txt").write_text(
        "0 0.10 0.10 0.40 0.10 0.40 0.40 0.10 0.40\n",
        encoding="utf-8",
    )
    proj = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    proj.sources = [OBBSource(path=str(source_root), name="preview_source")]

    dlg = TrainingDialog(proj)

    records = dlg._source_preview_records()
    assert records
    assert dlg.source_preview_status.text().startswith("Showing ")
    assert dlg.slice_group.preview.frame_size == (320, 240)
    assert dlg.slice_group.preview.frame_options == [(320, 240, 1), (640, 320, 1)]


# ---------------------------------------------------------------------------
# Augmentation group
# ---------------------------------------------------------------------------


def test_training_dialog_augmentation_roundtrip(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.aug_enabled = False
    proj.aug_fliplr = 0.0
    proj.aug_degrees = 45.0
    dlg = TrainingDialog(proj)
    assert dlg.aug_group.isChecked() is False
    assert abs(dlg.aug_fliplr.value() - 0.0) < 0.001
    assert abs(dlg.aug_degrees.value() - 45.0) < 0.001


# ---------------------------------------------------------------------------
# Publish group
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Loss plot
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Write-to-project round-trip
# ---------------------------------------------------------------------------


def test_training_dialog_write_to_project(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.sources[0].level = "polygon"
    dlg = TrainingDialog(proj)

    dlg.mode_combo.setCurrentIndex(dlg.mode_combo.findData("sequential"))
    dlg.task_combo.setCurrentIndex(dlg.task_combo.findData("segment"))
    dlg.spin_epochs.setValue(25)
    dlg.spin_seed.setValue(99)
    dlg.aug_group.setChecked(False)

    dlg._write_to_project()

    assert proj.training_mode == "sequential"
    assert proj.training_task == "segment"
    assert proj.role_seq_detect is True
    assert proj.role_seq_crop_segment is True
    assert proj.epochs == 25
    assert proj.seed == 99
    assert proj.aug_enabled is False


def test_training_dialog_on_done_persists_project_history(qapp, tmp_path, monkeypatch):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    captured = {}

    def fake_record(project, results):
        captured["project"] = project
        captured["results"] = results
        return [
            {
                "role": "obb_direct",
                "success": True,
                "project_model_path": str(tmp_path / "models" / "best.pt"),
            }
        ]

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.project.record_training_results",
        fake_record,
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.training_dialog.QMessageBox.information",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.training_dialog.QMessageBox.warning",
        lambda *args, **kwargs: None,
    )

    dlg._role_logs = {"obb_direct": ["line 1", "line 2"]}
    dlg._on_done([{"role": "obb_direct", "success": True, "artifact_path": ""}])

    assert captured["project"] is dlg._project
    assert captured["results"][0]["training_log"] == "line 1\nline 2"


def test_training_dialog_bounds_active_and_persisted_noisy_child_logs(
    qapp, tmp_path, monkeypatch
):
    from hydra_suite.detectkit.gui.dialogs import training_dialog as td

    dlg = td.TrainingDialog(_make_proj(tmp_path))
    dlg._current_role = "semantic_sam3"
    total = td.MAX_PERSISTED_ROLE_LOG_LINES + 250
    for index in range(total):
        dlg._append_log(f"noisy child record {index}:" + ("x" * 512))

    retained = dlg._role_logs["semantic_sam3"]
    assert len(retained) <= td.MAX_PERSISTED_ROLE_LOG_LINES
    assert sum(map(len, retained)) <= td.MAX_PERSISTED_ROLE_LOG_CHARS
    assert dlg._role_log_dropped["semantic_sam3"] > 0
    assert dlg.log_view.document().blockCount() <= td.MAX_VISIBLE_LOG_BLOCKS

    captured = {}
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.project.record_training_results",
        lambda _project, results: captured.setdefault("results", results),
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.training_dialog.QMessageBox.warning",
        lambda *_args, **_kwargs: None,
    )
    dlg._on_done([{"role": "semantic_sam3", "success": False}])

    persisted = captured["results"][0]["training_log"]
    assert "log retention dropped" in persisted
    assert len(persisted) <= td.MAX_PERSISTED_ROLE_LOG_CHARS + 128
