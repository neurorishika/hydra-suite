"""Tests for PoseKit's Smart Select dialog (ClassKit-style batch building)."""

from __future__ import annotations

import csv
import gc
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

from hydra_suite.posekit.core.selection import (  # noqa: E402
    REASON_ERROR,
    REASON_UNCERTAINTY,
)
from hydra_suite.posekit.gui.dialogs import smart_select as ss  # noqa: E402
from hydra_suite.posekit.gui.dialogs.smart_select import (  # noqa: E402
    PredictionSource,
    SmartSelectDialog,
)


@pytest.fixture()
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    store = {}
    monkeypatch.setattr(
        ss, "_load_dialog_settings", lambda key: dict(store.get(key, {}))
    )
    monkeypatch.setattr(
        ss, "_save_dialog_settings", lambda key, val: store.__setitem__(key, dict(val))
    )
    return store


@pytest.fixture(autouse=True)
def cleanup_qt_widgets(qapp):
    yield
    for widget in list(qapp.topLevelWidgets()):
        widget.close()
        widget.deleteLater()
    qapp.processEvents()
    gc.collect()


def _blobs(sizes, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(len(sizes), dim)) * 3
    return np.vstack(
        [
            centers[c] + rng.normal(scale=0.05, size=(n, dim))
            for c, n in enumerate(sizes)
        ]
    ).astype(np.float32)


def _dialog(tmp_path, names, labeled=(), frame_mode=False, source_ids=None, **kw):
    project = SimpleNamespace(
        enhance_enabled=False, out_root=tmp_path, keypoint_names=["a", "b"]
    )
    paths = [tmp_path / "images" / n for n in names]
    labeled_set = {paths[i] for i in labeled}
    return SmartSelectDialog(
        None,
        project,
        paths,
        lambda p: p in labeled_set,
        frame_mode=frame_mode,
        source_ids=source_ids,
        labeling_frames=set(labeled),
        **kw,
    )


def _build(qapp, dlg, emb, timeout=30.0):
    """Run Build Batch with precomputed embeddings and wait for the worker."""
    dlg._emb_all = emb
    dlg._emb_key = dlg._embedding_key()
    dlg._build()
    deadline = time.monotonic() + timeout
    while dlg._sel_worker is not None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    assert dlg._sel_worker is None, "selection worker did not finish"


def test_builds_ordered_batch_and_starts_labeling(qapp, tmp_path):
    names = [f"img{i:03d}.png" for i in range(60)]
    dlg = _dialog(tmp_path, names, labeled=range(10))
    dlg.n_spin.setValue(12)
    assert not dlg.btn_start.isEnabled()
    _build(qapp, dlg, _blobs([30, 25, 5]))

    assert len(dlg.batch_units) == 12 == dlg.table.rowCount()
    assert not set(dlg.batch_units) & set(range(10))  # labeled are not candidates
    assert dlg.btn_start.isEnabled()
    # Without predictions the model slots are off and say why.
    assert not dlg.error_spin.isEnabled()
    assert "Model signals off" in dlg.model_label.text()
    assert dlg.summary_label.text().startswith("<b>12 images")

    dlg._finish(start=True)
    assert dlg.result() == QDialog.Accepted
    assert dlg.start_labeling is True
    assert dlg.selected_indices == dlg.batch_units  # pick order kept


def test_clusters_csv_covers_every_frame(qapp, tmp_path):
    names = [f"img{i:03d}.png" for i in range(40)]
    dlg = _dialog(tmp_path, names, labeled=range(5))
    _build(qapp, dlg, _blobs([20, 20]))
    csv_path = tmp_path / "posekit" / "clusters" / "clusters.csv"
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 40
    assert {r["image"] for r in rows} == {str(tmp_path / "images" / n) for n in names}


def test_remove_rows_and_explorer_merge(qapp, tmp_path, monkeypatch):
    names = [f"img{i:03d}.png" for i in range(40)]
    dlg = _dialog(tmp_path, names)
    dlg.n_spin.setValue(6)
    _build(qapp, dlg, _blobs([20, 20]))
    first = list(dlg.batch_units)

    dlg.table.selectRow(0)
    dlg._remove_selected_rows()
    assert dlg.batch_units == first[1:]

    new_unit = next(u for u in range(40) if u not in first)

    class _FakeExplorer:
        def __init__(self, **kwargs):
            assert kwargs["initial_selection"] == first[1:]
            _FakeExplorer.kwargs = kwargs
            self.selected_indices = [first[2], new_unit]  # drop all but one

        def exec(self):
            return QDialog.Accepted

    monkeypatch.setattr(
        "hydra_suite.posekit.gui.dialogs.exploration.EmbeddingExplorerDialog",
        _FakeExplorer,
    )
    dlg._open_explorer()
    assert dlg.batch_units == [first[2], new_unit]
    assert dlg._unit_reason[new_unit] == ss.REASON_EXPLORER
    assert "Picked in Explorer" in dlg.table.item(1, 3).text()


def test_frame_mode_picks_whole_frames(qapp, tmp_path):
    # Two instances per source frame (did = frame * 10000 + instance).
    names = [f"did{f * 10000 + k}.jpg" for f in range(1, 21) for k in range(2)]
    dlg = _dialog(tmp_path, names, frame_mode=True, source_ids=["src"] * len(names))
    dlg.n_spin.setValue(5)
    emb = np.repeat(_blobs([10, 10]), 2, axis=0)
    _build(qapp, dlg, emb)
    assert len(dlg.batch_units) == 5
    picked = dlg.batch_instances()
    assert len(picked) == 10
    frames = {names[i][3:].split(".")[0] for i in picked}
    assert len({int(f) // 10000 for f in frames}) == 5
    assert "(+1 on frame)" in dlg.table.item(0, 1).text()


def test_scope_labeling_set_reorders_existing_frames(qapp, tmp_path):
    names = [f"img{i:03d}.png" for i in range(30)]
    dlg = _dialog(tmp_path, names)
    dlg.labeling_frames = {3, 4, 5, 20}
    dlg.cb_scope.setCurrentText("Labeling set")
    dlg.n_spin.setValue(10)
    _build(qapp, dlg, _blobs([15, 15]))
    assert sorted(dlg.batch_units) == [3, 4, 5, 20]


def _write_png(path, size=(64, 64)):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(path)


def test_model_signals_drive_error_and_uncertainty_picks(qapp, tmp_path):
    names = [f"img{i:03d}.png" for i in range(40)]
    paths = [tmp_path / "images" / n for n in names]
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    labeled = list(range(0, 4)) + list(range(20, 24))
    for i in labeled:
        _write_png(paths[i])
        # bbox covers the image; keypoints at (0.25, 0.25) and (0.75, 0.75)
        (labels_dir / f"{paths[i].stem}.txt").write_text(
            "0 0.5 0.5 1.0 1.0 0.25 0.25 2 0.75 0.75 2\n"
        )
    preds = {}
    for i in range(40):
        conf = 0.9 if i % 2 else 0.2  # even frames: model unsure
        preds[str(paths[i])] = [(16.0, 16.0, conf), (48.0, 48.0, conf)]
    for i in range(20, 24):  # blob 1 labels: the model is far off
        preds[str(paths[i])] = [(60.0, 2.0, 0.9), (2.0, 60.0, 0.9)]

    dlg = _dialog(
        tmp_path,
        names,
        labeled=labeled,
        label_path_fn=lambda p: labels_dir / f"{p.stem}.txt",
        prediction_source=PredictionSource("m.pt (yolo)", preds),
    )
    assert dlg.error_spin.isEnabled()
    assert "40 of 40 frames" in dlg.model_label.text()
    dlg.n_spin.setValue(10)
    _build(qapp, dlg, _blobs([20, 20]))

    out = dlg._outcome
    assert out.n_predicted == 40 and out.n_error_scored == 8
    assert out.result.info["n_errors"] == 4
    err_picks = out.result.breakdown[REASON_ERROR]
    assert err_picks and all(20 <= u < 40 for u in err_picks)
    unc_picks = out.result.breakdown[REASON_UNCERTAINTY]
    assert unc_picks and all(u % 2 == 0 for u in unc_picks)
    row = dlg.batch_units.index(err_picks[0])
    assert dlg.table.item(row, 4).text().startswith("looks like img02")
    assert "with predictions" in dlg.lbl_status.text()


def test_settings_persist_on_any_close(qapp, tmp_path, isolated_settings):
    dlg = _dialog(tmp_path, ["a.png", "b.png"])
    dlg.n_spin.setValue(77)
    dlg.rare_spin.setValue(33)
    dlg.cluster_method_combo.setCurrentIndex(1)
    dlg.reject()
    saved = isolated_settings["smart_select"]
    assert saved["n"] == 77 and saved["mix"]["rare_spin"] == 33
    assert saved["cluster_method_v2"] == "minibatch"

    again = _dialog(tmp_path, ["a.png", "b.png"])
    assert again.n_spin.value() == 77 and again.rare_spin.value() == 33
    assert again.cluster_method_combo.currentData() == "minibatch"


def test_mix_remainder_label(qapp, tmp_path):
    dlg = _dialog(tmp_path, ["a.png"])
    assert dlg.remainder_label.text() == "Random checks: 5%"
    dlg.diversity_spin.setValue(90)
    assert "scaled down" in dlg.remainder_label.text()


def test_nothing_eligible_shows_message(qapp, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(
        ss.QMessageBox, "information", staticmethod(lambda *a, **k: shown.append(a[2]))
    )
    dlg = _dialog(tmp_path, ["a.png", "b.png"], labeled=[0, 1])
    dlg._build()
    assert shown and "No frames match" in shown[0]
    assert dlg._sel_worker is None


def test_old_import_path_still_works():
    from hydra_suite.posekit.gui.dialogs.exploration import SmartSelectDialog as Old

    assert Old is SmartSelectDialog


def test_without_predictions_model_share_fills_gaps_not_random(qapp, tmp_path):
    names = [f"img{i:03d}.png" for i in range(80)]
    dlg = _dialog(tmp_path, names, labeled=range(10))
    dlg.n_spin.setValue(20)
    _build(qapp, dlg, _blobs([40, 30, 10]))
    bd = dlg._outcome.result.breakdown
    # Default mix: 5% random checks -> 1 of 20; errors+uncertainty (45%) go
    # to the diversity top-up.
    assert len(bd.get(ss.REASON_AUDIT, [])) == 1
    assert len(bd[ss.REASON_DIVERSITY]) >= 6 + 9


def _slow_selection(monkeypatch, delay=0.5):
    import hydra_suite.posekit.core.smart_select as job

    real = job.run_smart_select
    calls = []

    def slow(req, status=lambda s: None, is_canceled=lambda: False):
        calls.append(1)
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            if is_canceled():
                raise job.SmartSelectCanceled()
            time.sleep(0.01)
        return real(req, status, is_canceled)

    monkeypatch.setattr(job, "run_smart_select", slow)
    return calls


def _pump_until(qapp, cond, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


def test_actions_are_ignored_while_busy(qapp, tmp_path, monkeypatch):
    names = [f"img{i:03d}.png" for i in range(40)]
    dlg = _dialog(tmp_path, names)
    dlg.n_spin.setValue(6)
    _build(qapp, dlg, _blobs([20, 20]))
    first = list(dlg.batch_units)

    calls = _slow_selection(monkeypatch)
    dlg._build()  # second build, now running
    assert dlg._busy and not dlg.btn_build.isEnabled()
    dlg.table.selectRow(0)
    dlg._remove_selected_rows()  # Delete key while busy
    assert dlg.batch_units == first
    assert not dlg.btn_build.isEnabled() and dlg.btn_cancel.isEnabled()
    dlg._build()  # double build is ignored
    dlg._finish(start=True)  # can't accept a batch mid-build
    assert dlg.result() != QDialog.Accepted
    _pump_until(qapp, lambda: dlg._sel_worker is None)
    assert calls == [1]
    assert not dlg._busy and dlg.btn_build.isEnabled()


def test_cancel_after_embeddings_skips_the_selection(qapp, tmp_path, monkeypatch):
    names = [f"img{i:03d}.png" for i in range(30)]
    dlg = _dialog(tmp_path, names)
    started = []
    monkeypatch.setattr(dlg, "_start_selection", lambda: started.append(1))
    dlg._eligible_indices = list(range(30))
    dlg._set_busy(True)
    dlg._emb_key_pending = dlg._embedding_key()
    dlg._cancel()  # user canceled; the worker had already passed its checks
    dlg._on_emb_finished(_blobs([15, 15]), dlg._eligible_indices, {})
    assert started == []
    assert dlg._emb_all is not None  # kept for the next Build
    assert dlg.lbl_status.text() == "Canceled." and not dlg._busy


def test_cancel_during_selection(qapp, tmp_path, monkeypatch):
    names = [f"img{i:03d}.png" for i in range(30)]
    dlg = _dialog(tmp_path, names)
    _slow_selection(monkeypatch, delay=5)
    dlg._emb_all = _blobs([15, 15])
    dlg._emb_key = dlg._embedding_key()
    dlg._build()
    dlg._cancel()
    _pump_until(qapp, lambda: dlg._sel_worker is None)
    assert dlg._outcome is None and dlg.batch_units == []
    assert dlg.lbl_status.text() == "Canceled." and not dlg._busy


def test_close_while_building_does_not_abort(qapp, tmp_path, monkeypatch):
    names = [f"img{i:03d}.png" for i in range(30)]
    dlg = _dialog(tmp_path, names)
    _slow_selection(monkeypatch, delay=1)
    dlg._emb_all = _blobs([15, 15])
    dlg._emb_key = dlg._embedding_key()
    dlg._build()
    dlg.reject()
    dlg.deleteLater()
    qapp.processEvents()
    _pump_until(qapp, lambda: not ss._ORPHANED_THREADS, timeout=10)
    assert not ss._ORPHANED_THREADS


@pytest.mark.parametrize(
    "bad",
    [
        {"n": None},
        {"n": "abc"},
        {"mix": 5},
        {"mix": {"error_spin": None}},
        {"rare_alpha": None, "dedup": "yes", "model": 3, "device": "tpu"},
        ["not", "a", "dict"],
    ],
)
def test_corrupt_settings_fall_back_to_defaults(qapp, tmp_path, monkeypatch, bad):
    monkeypatch.setattr(ss, "_load_dialog_settings", lambda key: bad)
    dlg = _dialog(tmp_path, ["a.png"])
    assert dlg.n_spin.value() == 50 and dlg.error_spin.value() == 20


def test_auto_cluster_count_is_shown(qapp, tmp_path):
    names = [f"img{i:03d}.png" for i in range(400)]
    dlg = _dialog(tmp_path, names)
    assert dlg.cb_auto_k.isChecked() and dlg.k_spin.value() == 20  # sqrt(400)
    dlg.cb_auto_k.setChecked(False)
    assert dlg.k_spin.isEnabled() and dlg.k_spin.value() == 20


def test_legacy_embedding_cache_is_removed(qapp, tmp_path):
    legacy = tmp_path / "posekit" / "embeddings" / "embeddings" / "abc"
    legacy.mkdir(parents=True)
    (legacy / "embeddings.npy").write_bytes(b"x")
    keep = tmp_path / "posekit" / "embeddings" / "current"
    keep.mkdir(parents=True)
    dlg = _dialog(tmp_path, ["a.png"])
    dlg._remove_legacy_embedding_cache()
    assert not legacy.parent.exists() and keep.exists()


def test_frame_mode_banner_admits_ungrouped_names(qapp, tmp_path):
    from PySide6.QtWidgets import QLabel

    dlg = _dialog(
        tmp_path,
        ["vid__f000644__000.png", "vid__f000644__001.png"],
        frame_mode=True,
        source_ids=["s", "s"],
    )
    texts = [w.text() for w in dlg.findChildren(QLabel)]
    assert any("counts as its own frame" in t for t in texts)


def test_frame_mode_marks_partly_labeled_frames(qapp, tmp_path):
    names = [f"did{f * 10000 + k}.jpg" for f in range(1, 21) for k in range(2)]
    dlg = _dialog(
        tmp_path,
        names,
        labeled=[0],  # frame 1 has one of its two individuals labeled
        frame_mode=True,
        source_ids=["src"] * len(names),
    )
    dlg.cb_scope.setCurrentText("All frames")
    dlg.cb_exclude_in_labeling.setChecked(False)
    dlg.n_spin.setValue(20)
    _build(qapp, dlg, np.repeat(_blobs([10, 10]), 2, axis=0))
    row = dlg.batch_units.index(0)
    assert "(1 of 2 already labeled)" in dlg.table.item(row, 4).text()


def test_explorer_cannot_add_frames_outside_scope(qapp, tmp_path, monkeypatch):
    names = [f"img{i:03d}.png" for i in range(40)]
    dlg = _dialog(tmp_path, names, labeled=[0, 1])
    dlg.n_spin.setValue(4)
    _build(qapp, dlg, _blobs([20, 20]))
    batch = list(dlg.batch_units)

    class _Explorer:
        def __init__(self, **kwargs):
            self.selected_indices = batch + [0]  # 0 is labeled

        def exec(self):
            return QDialog.Accepted

    monkeypatch.setattr(
        "hydra_suite.posekit.gui.dialogs.exploration.EmbeddingExplorerDialog", _Explorer
    )
    dlg._open_explorer()
    assert dlg.batch_units == batch
    assert "Skipped 1" in dlg.lbl_status.text()
