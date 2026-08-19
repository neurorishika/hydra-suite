"""Shared-trunk (`multihead_custom_shared`) must be a first-class ClassKit path.

A shared-trunk model is ONE .pth whose forward emits the *concatenated* per-factor
logits (width = sum of factor label counts). v2 multi-head checkpoints deliberately
omit a top-level ``class_names``, so any consumer that falls back to ``self.classes``
(which is only ``scheme.factors[0].labels``) mislabels predictions as ``pred_N``.

These tests pin the contract: every ClassKit entry point must produce the same
``(probs, class_names, prediction_heads)`` state the N-artifact multi-head path
already produces, and TrackerKit must consume the same artifact per-factor.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtWidgets import QApplication  # noqa: E402

CNPF = [["red", "yellow"], ["small", "big", "huge"]]
FACTORS = ["color", "size"]
FLAT_NAMES = ["red", "yellow", "small", "big", "huge"]
TOTAL_WIDTH = 5


@pytest.fixture()
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture()
def shared_trunk_ckpt(tmp_path: Path) -> Path:
    from hydra_suite.training.multihead_torchvision_model import (
        build_multihead_torchvision_classifier,
    )
    from hydra_suite.training.torchvision_model import save_torchvision_checkpoint

    model = build_multihead_torchvision_classifier(
        backbone="resnet18",
        class_names_per_factor=CNPF,
        trainable_layers=-1,
        head_hidden_dim=16,
        head_dropout=0.0,
        input_size=64,
    )
    path = tmp_path / "shared_trunk.pth"
    save_torchvision_checkpoint(
        model=model,
        backbone="resnet18",
        class_names=[],
        factor_names=FACTORS,
        class_names_per_factor=CNPF,
        input_size=(64, 64),
        best_val_acc=0.5,
        history={"train_loss": [], "val_acc": []},
        trainable_layers=-1,
        backbone_lr_scale=0.1,
        monochrome=False,
        extra_meta={
            "head_kind": "multihead_shared_trunk",
            "head_hidden_dim": 16,
            "head_dropout": 0.0,
        },
        path=path,
    )
    return path


@pytest.fixture()
def sample_images(tmp_path: Path) -> list[Path]:
    from PIL import Image

    paths = []
    for i in range(3):
        p = tmp_path / f"img{i}.png"
        Image.fromarray(np.full((64, 64, 3), 40 * i + 10, dtype=np.uint8)).save(p)
        paths.append(p)
    return paths


def _scheme():
    return types.SimpleNamespace(
        factors=[
            types.SimpleNamespace(name="color", labels=["red", "yellow"]),
            types.SimpleNamespace(name="size", labels=["small", "big", "huge"]),
        ]
    )


def _stub_window(images, monkeypatch):
    """A MainWindow stub that borrows the real prediction-state implementations."""
    from hydra_suite.classkit.gui.main_window import MainWindow

    def _borrow(name):
        raw = MainWindow.__dict__.get(name)
        if isinstance(raw, staticmethod):
            return staticmethod(raw.__func__)
        return getattr(MainWindow, name)

    borrowed = {
        name: _borrow(name)
        for name in (
            "_set_model_prediction_state",
            "_normalize_prediction_heads",
            "_derive_prediction_heads_from_scheme",
            "_current_prediction_heads",
            "_prediction_labels_for_plot",
            "_prediction_label_from_head_slice",
            "_prediction_confidences_from_probs",
            "_normalize_prediction_confidence_threshold",
            "_threshold_prediction_label",
            "_prediction_summary_for_index",
            "_run_shared_trunk_inference",
            "is_shared_trunk_checkpoint",
            "_load_model_from_cache_entry",
            "_load_custom_cnn_checkpoint",
            "_run_post_training_inference",
        )
        if name in MainWindow.__dict__ or hasattr(MainWindow, name)
    }

    class Stub:
        locals().update(borrowed)

        def __init__(self):
            self.image_paths = list(images)
            # ClassKit sets self.classes = scheme.factors[0].labels
            self.classes = ["red", "yellow"]
            self._model_probs = None
            self._model_class_names = []
            self._active_model_mode = ""
            self._model_prediction_heads = None
            self._model_prediction_confidence_threshold = None
            self.image_confidences = []
            self.umap_model_coords = None
            self.pca_model_coords = None
            self._show_model_umap = False
            self._show_model_pca = False
            self._last_training_settings = {}
            self.persisted = []
            self.btn_umap_embedding = types.SimpleNamespace(setChecked=lambda *_: None)
            self.btn_umap_model = types.SimpleNamespace(setChecked=lambda *_: None)
            self.status = types.SimpleNamespace(showMessage=lambda *a, **k: None)
            self.progress_bar = types.SimpleNamespace(
                setValue=lambda *_: None, setVisible=lambda *_: None
            )

        def _current_prediction_confidence_threshold(self):
            return 0.0

        def _resolve_training_scheme(self):
            return _scheme()

        def _resolve_classifier_device(self, _path):
            return "cpu"

        def _set_model_projection_buttons_enabled(self, *_a):
            pass

        def _update_al_status(self):
            pass

        def update_explorer_plot(self, *a, **k):
            pass

        def _evaluate_model_on_labeled(self, *a, **k):
            pass

        def _replot_umap_model_space(self, *a, **k):
            pass

        def _set_active_evaluation_from_meta(self, *a, **k):
            pass

        def _set_heldout_validation_summary(self, *a, **k):
            pass

        def _cached_model_evaluation_info(self, *a, **k):
            return None

        def _cached_model_class_names(self, *a, **k):
            return None

        def _cached_model_training_settings(self, *a, **k):
            return None

        def _resolve_prediction_confidence_threshold_for_path(self, *a, **k):
            return None

        def _validation_summary_from_value(self, *a, **k):
            return None

        def _validation_summary_from_results(self, *a, **k):
            return None

        def _set_active_evaluation_selection(self, *a, **k):
            pass

        def _persist_prediction_cache(self, probs, class_names, mode, **kwargs):
            self.persisted.append(
                {
                    "class_names": list(class_names),
                    "mode": mode,
                    "prediction_heads": kwargs.get("prediction_heads"),
                }
            )

        def _threadpool_start(self, worker):
            worker.run()  # synchronous for tests

    return Stub()


# ── the worker contract ───────────────────────────────────────────────────


def test_shared_trunk_worker_emits_per_factor_names_and_heads(
    qapp, shared_trunk_ckpt, sample_images
):
    """The worker must report per-factor names + head slices, not a flat list."""
    from hydra_suite.classkit.jobs.task_workers import SharedTrunkInferenceWorker

    captured = {}
    worker = SharedTrunkInferenceWorker(
        shared_trunk_ckpt, sample_images, device="cpu", batch_size=2
    )
    worker.signals.success.connect(lambda r: captured.update(r))
    worker.run()

    probs = np.asarray(captured["probs"])
    assert probs.shape == (3, TOTAL_WIDTH)
    assert list(captured["class_names"]) == FLAT_NAMES
    assert captured["prediction_heads"] == [
        {"factor": "color", "class_names": ["red", "yellow"], "start": 0, "end": 2},
        {
            "factor": "size",
            "class_names": ["small", "big", "huge"],
            "start": 2,
            "end": 5,
        },
    ]


def test_shared_trunk_worker_softmaxes_per_head_not_across_all_columns(
    qapp, shared_trunk_ckpt, sample_images
):
    """Each head's probabilities must sum to 1 independently."""
    from hydra_suite.classkit.jobs.task_workers import SharedTrunkInferenceWorker

    captured = {}
    worker = SharedTrunkInferenceWorker(
        shared_trunk_ckpt, sample_images, device="cpu", batch_size=8
    )
    worker.signals.success.connect(lambda r: captured.update(r))
    worker.run()

    probs = np.asarray(captured["probs"])
    np.testing.assert_allclose(probs[:, 0:2].sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(probs[:, 2:5].sum(axis=1), 1.0, atol=1e-5)


# ── ClassKit entry points ─────────────────────────────────────────────────


def test_model_history_load_of_shared_trunk_labels_every_column(
    qapp, shared_trunk_ckpt, sample_images, monkeypatch
):
    """Loading from Model History must never yield a `pred_N` placeholder."""
    from hydra_suite.classkit.gui.main_window import MainWindow

    win = _stub_window(sample_images, monkeypatch)
    entry = {
        "mode": "multihead_custom_shared",
        "artifact_paths": [str(shared_trunk_ckpt)],
        "class_names": ["red", "yellow"],
        "meta": {},
    }
    MainWindow._load_model_from_cache_entry(win, entry)

    assert np.asarray(win._model_probs).shape == (3, TOTAL_WIDTH)
    assert list(win._model_class_names) == FLAT_NAMES
    heads = win._model_prediction_heads
    assert [h["factor"] for h in heads] == FACTORS
    labels = win._prediction_labels_for_plot()
    assert all("pred_" not in str(lbl) for lbl in labels), labels
    # composite label = exactly one pick per factor, joined by MULTIHEAD_LABEL_SEPARATOR
    expected = {f"{a}_{b}" for a in CNPF[0] for b in CNPF[1]}
    assert {str(lbl) for lbl in labels} <= expected, labels


def test_checkpoint_load_dispatches_shared_trunk_by_head_kind(
    qapp, shared_trunk_ckpt, sample_images, monkeypatch
):
    """The generic 'load a .pth' path must detect head_kind, not treat it as flat."""
    import torch

    from hydra_suite.classkit.gui.main_window import MainWindow

    win = _stub_window(sample_images, monkeypatch)
    ckpt = torch.load(str(shared_trunk_ckpt), map_location="cpu", weights_only=False)
    MainWindow._load_custom_cnn_checkpoint(
        win, shared_trunk_ckpt, ckpt, show_message_box=False
    )

    assert list(win._model_class_names) == FLAT_NAMES
    assert [h["factor"] for h in win._model_prediction_heads] == FACTORS
    assert all("pred_" not in str(x) for x in win._prediction_labels_for_plot())


def test_shared_trunk_prediction_cache_persists_head_metadata(
    qapp, shared_trunk_ckpt, sample_images, monkeypatch
):
    """The DB cache must store per-factor names so a restart restores real labels."""
    from hydra_suite.classkit.gui.main_window import MainWindow

    win = _stub_window(sample_images, monkeypatch)
    entry = {
        "mode": "multihead_custom_shared",
        "artifact_paths": [str(shared_trunk_ckpt)],
        "class_names": ["red", "yellow"],
        "meta": {},
    }
    MainWindow._load_model_from_cache_entry(win, entry)

    assert win.persisted, "prediction cache was never written"
    cached = win.persisted[-1]
    assert list(cached["class_names"]) == FLAT_NAMES
    assert [h["factor"] for h in cached["prediction_heads"]] == FACTORS


def test_post_training_preview_no_longer_skips_shared_trunk(
    qapp, shared_trunk_ckpt, sample_images, monkeypatch
):
    """Training a shared trunk must produce an in-dialog preview like every other mode."""
    from hydra_suite.classkit.gui.main_window import MainWindow

    win = _stub_window(sample_images, monkeypatch)
    logs: list[str] = []
    dialog = types.SimpleNamespace(append_log=logs.append)

    MainWindow._run_post_training_inference(
        win,
        [{"artifact_path": str(shared_trunk_ckpt)}],
        False,  # is_yolo
        False,  # multi_head (shared trunk publishes as a single artifact)
        ["red", "yellow"],
        dialog,
        lambda _result: None,
    )

    assert not any("skipping in-dialog preview" in line for line in logs), logs
    assert list(win._model_class_names) == FLAT_NAMES
    assert [h["factor"] for h in win._model_prediction_heads] == FACTORS


# ── TrackerKit consumption ────────────────────────────────────────────────


def test_trackerkit_cnn_identity_reads_shared_trunk_per_factor(shared_trunk_ckpt):
    """TrackerKit's CNN identity backend must yield one class per factor."""
    from hydra_suite.core.individual.classification.cnn import CNNIdentityBackend
    from hydra_suite.runtime.resolver import ResolvedBackend

    config = types.SimpleNamespace(
        model_path=str(shared_trunk_ckpt),
        confidence=0.0,
        scoring_mode="atomic",
    )
    clf = CNNIdentityBackend(config, resolved=ResolvedBackend("torch", "cpu", False))
    crops = [np.full((64, 64, 3), 30 * i + 10, dtype=np.uint8) for i in range(2)]
    preds = clf.predict_batch(crops)

    assert len(preds) == 2
    for pred in preds:
        assert tuple(pred.factor_names) == tuple(FACTORS)
        assert pred.class_names[0] in CNPF[0]
        assert pred.class_names[1] in CNPF[1]


def test_trackerkit_import_dialog_presents_shared_trunk_as_multihead(
    qapp, shared_trunk_ckpt
):
    """The TrackerKit import preview must show both factors and offer scoring mode."""
    from hydra_suite.trackerkit.gui.dialogs.cnn_identity_import_dialog import (
        CNNIdentityImportDialog,
        describe_cnn_identity_candidate,
    )

    summary = describe_cnn_identity_candidate(str(shared_trunk_ckpt))
    assert summary["is_multihead"] is True
    assert list(summary["factor_names"]) == FACTORS
    assert [list(c) for c in summary["class_names_per_factor"]] == CNPF

    dialog = CNNIdentityImportDialog(summary)
    try:
        # multi-head models must expose the scoring-mode choice
        assert dialog.scoring_mode() == "atomic"
        assert dialog._per_head_btn is not None
        dialog._per_head_btn.setChecked(True)
        assert dialog.scoring_mode() == "per_head_average"
    finally:
        dialog.close()
        dialog.deleteLater()


def test_shared_trunk_publishes_with_per_factor_metadata(shared_trunk_ckpt):
    """Publishing must stamp per-factor metadata TrackerKit can consume."""
    from hydra_suite.training.model_publish import classifier_metadata_for_artifact

    meta = classifier_metadata_for_artifact(str(shared_trunk_ckpt))
    assert list(meta["factor_names"]) == FACTORS
    assert [list(c) for c in meta["class_names_per_factor"]] == CNPF
    # v2 multi-head artifacts must never carry a flat class list
    assert "class_names" not in meta or not meta.get("class_names")


def test_classkit_preview_matches_trackerkit_inference_on_same_crop(
    qapp, shared_trunk_ckpt, sample_images
):
    """ClassKit's shared-trunk preview must agree with TrackerKit's inference.

    Guards the preprocessing contract (canonical fit + RGB channel order): a
    BGR/RGB flip or a different normalization would silently diverge the two
    kits' predictions for the same image.
    """
    from PIL import Image

    from hydra_suite.classkit.jobs.task_workers import SharedTrunkInferenceWorker
    from hydra_suite.core.individual.classification.backend import ClassifierBackend
    from hydra_suite.runtime.resolver import ResolvedBackend
    from hydra_suite.training.canonical_transform import CanonicalFitTransform

    captured = {}
    worker = SharedTrunkInferenceWorker(
        shared_trunk_ckpt, sample_images, device="cpu", batch_size=8
    )
    worker.signals.success.connect(lambda r: captured.update(r))
    worker.run()
    classkit_probs = np.asarray(captured["probs"])

    fit = CanonicalFitTransform((64, 64))
    crops = [
        fit(np.asarray(Image.open(str(p)).convert("RGB"), dtype=np.uint8))
        for p in sample_images
    ]
    backend = ClassifierBackend(
        str(shared_trunk_ckpt), ResolvedBackend("torch", "cpu", False)
    )
    try:
        per_crop = backend.predict_batch(crops, input_is_bgr=False)
    finally:
        backend.close()
    trackerkit_probs = np.stack(
        [np.concatenate([np.asarray(p) for p in factors]) for factors in per_crop]
    )

    np.testing.assert_allclose(classkit_probs, trackerkit_probs, rtol=0, atol=1e-6)


def test_shared_trunk_survives_a_stale_recorded_mode(
    qapp, shared_trunk_ckpt, sample_images, monkeypatch
):
    """A shared-trunk artifact filed under a stale mode must not fall back to flat."""
    from hydra_suite.classkit.gui.main_window import MainWindow

    win = _stub_window(sample_images, monkeypatch)
    entry = {
        "mode": "flat_custom",  # stale/incorrect mode on the history row
        "artifact_paths": [str(shared_trunk_ckpt)],
        "class_names": ["red", "yellow"],
        "meta": {},
    }
    MainWindow._load_model_from_cache_entry(win, entry)

    assert list(win._model_class_names) == FLAT_NAMES
    assert [h["factor"] for h in win._model_prediction_heads] == FACTORS
    assert all("pred_" not in str(x) for x in win._prediction_labels_for_plot())


def test_al_candidate_detail_reports_composite_prediction(
    qapp, shared_trunk_ckpt, sample_images, monkeypatch
):
    """The AL candidate row must show the composite prediction, not one head's argmax.

    A global argmax over the concatenated columns would report whichever single
    factor happened to have the highest probability.
    """
    from hydra_suite.classkit.gui.main_window import MainWindow

    win = _stub_window(sample_images, monkeypatch)
    MainWindow._load_model_from_cache_entry(
        win,
        {
            "mode": "multihead_custom_shared",
            "artifact_paths": [str(shared_trunk_ckpt)],
            "class_names": ["red", "yellow"],
            "meta": {},
        },
    )

    summary = MainWindow._prediction_summary_for_index(win, 0, top_k=1)
    predicted = str(summary["predicted_label"])
    expected = {f"{a}_{b}" for a in CNPF[0] for b in CNPF[1]}
    assert predicted in expected, predicted
    # both factors contribute -- not a bare single-factor label
    assert predicted not in CNPF[0] and predicted not in CNPF[1]
