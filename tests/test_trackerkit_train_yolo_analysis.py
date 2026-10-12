"""TrackerKit training center: dataset analysis must not block the GUI."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture(autouse=True)
def _hermetic_dirs(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("hydra_home")
    monkeypatch.setenv("HYDRA_DATA_DIR", str(home / "data"))
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(home / "config"))


def _write_obb_dataset(root: Path) -> Path:
    images_dir = root / "images" / "train"
    labels_dir = root / "labels" / "train"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    image = QImage(320, 240, QImage.Format.Format_RGB32)
    image.fill(0)
    assert image.save(str(images_dir / "sample.png"))
    (labels_dir / "sample.txt").write_text(
        "0 0.10 0.10 0.40 0.10 0.40 0.40 0.10 0.40\n", encoding="utf-8"
    )
    (root / "dataset.yaml").write_text(
        "train: images/train\nval: images/train\nnames:\n  0: ant\n",
        encoding="utf-8",
    )
    return root


def _wait_until(qapp, predicate, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "condition not met before timeout"
        qapp.processEvents()
        time.sleep(0.01)


def test_analyze_and_preview_runs_off_the_gui_thread(qapp, tmp_path, monkeypatch):
    from PySide6.QtCore import QThread

    import hydra_suite.training.dataset_inspector as inspector
    from hydra_suite.trackerkit.gui.dialogs.train_yolo_dialog import TrainYoloDialog

    source = _write_obb_dataset(tmp_path / "src")
    release = threading.Event()
    seen: dict[str, object] = {}
    real_analyze = inspector.analyze_obb_sizes

    def blocking_analyze(*args, **kwargs):
        seen["thread"] = QThread.currentThread()
        assert release.wait(10.0)
        return real_analyze(*args, **kwargs)

    monkeypatch.setattr(inspector, "analyze_obb_sizes", blocking_analyze)

    dlg = TrainYoloDialog()
    try:
        dlg._add_source_row("obb", str(source))
        dlg._analyze_and_preview()

        assert not release.is_set()
        assert "Analyzing" in dlg.validation_view.toPlainText()
        assert not dlg.btn_analyze.isEnabled()

        release.set()
        _wait_until(qapp, lambda: dlg._analysis_worker is None)

        assert seen["thread"] is not qapp.thread()
        assert "DATASET SIZE ANALYSIS" in dlg.validation_view.toPlainText()
        assert dlg.preview_grid.count() > 0
        assert dlg.btn_analyze.isEnabled()
    finally:
        release.set()
        dlg.reject()


def test_closing_dialog_cancels_running_analysis(qapp, tmp_path, monkeypatch):
    import hydra_suite.training.dataset_inspector as inspector
    from hydra_suite.trackerkit.gui.dialogs.train_yolo_dialog import TrainYoloDialog

    source = _write_obb_dataset(tmp_path / "src")
    entered = threading.Event()
    cancel_seen = threading.Event()

    def cancellable_analyze(*_args, should_cancel=None, **_kwargs):
        entered.set()
        for _ in range(500):
            if should_cancel is not None and should_cancel():
                cancel_seen.set()
                break
            threading.Event().wait(0.01)
        return inspector.OBBSizeStats()

    monkeypatch.setattr(inspector, "analyze_obb_sizes", cancellable_analyze)

    dlg = TrainYoloDialog()
    dlg._add_source_row("obb", str(source))
    dlg._analyze_and_preview()
    worker = dlg._analysis_worker
    assert worker is not None
    assert entered.wait(5.0)

    dlg.reject()

    assert cancel_seen.wait(5.0)
    assert worker.wait(5000)
    assert dlg._analysis_worker is None
