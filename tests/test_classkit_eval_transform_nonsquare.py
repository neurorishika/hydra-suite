"""F7c: TorchvisionInferenceWorker._build_transform must fit crops into the
true (input_h, input_w) shape for non-square classifiers, matching how
training (training/runner.py) derives (H, W) -- not force them square.

Before the fix, _build_transform always did
    sz = self.input_size; CanonicalFitTransform((sz, sz))
which silently squished non-square classifiers at eval time even though
training and inference both preserve the true (H, W).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")

task_workers = pytest.importorskip("hydra_suite.classkit.jobs.task_workers")
TorchvisionInferenceWorker = task_workers.TorchvisionInferenceWorker


def _make_worker(input_size):
    return TorchvisionInferenceWorker(
        model_path=Path("/tmp/fake_model.pth"),
        image_paths=[],
        class_names=["left"],
        input_size=input_size,
    )


def test_build_transform_uses_true_hw_for_nonsquare_input_size():
    input_h, input_w = 64, 128
    worker = _make_worker((input_h, input_w))

    # The transform pipeline's first callable step wraps a CanonicalFitTransform
    # closed over in `_fit`; recover it via the closure cells instead of
    # re-deriving from self.input_size, so the assertion is on what the
    # transform *actually built*, not a restatement of the input.
    transform = worker._build_transform()
    fit_step = transform.transforms[0]
    closure = {
        var: cell.cell_contents
        for var, cell in zip(fit_step.__code__.co_freevars, fit_step.__closure__)
    }
    fit_transform = closure["fit_transform"]

    assert fit_transform.model_hw == (input_h, input_w)
    assert fit_transform.model_hw != (input_w, input_h)  # not swapped
    assert fit_transform.model_hw[0] != fit_transform.model_hw[1]  # truly non-square


def test_build_transform_actually_fits_nonsquare_output_shape():
    pytest.importorskip("torch")

    input_h, input_w = 64, 128
    worker = _make_worker((input_h, input_w))
    transform = worker._build_transform()

    img = Image.fromarray(
        (np.random.rand(50, 50, 3) * 255).astype(np.uint8), mode="RGB"
    )
    out = transform(img)

    assert tuple(out.shape) == (3, input_h, input_w)


def test_build_transform_square_int_input_size_still_square():
    worker = _make_worker(8)
    transform = worker._build_transform()
    fit_step = transform.transforms[0]
    closure = {
        var: cell.cell_contents
        for var, cell in zip(fit_step.__code__.co_freevars, fit_step.__closure__)
    }
    assert closure["fit_transform"].model_hw == (8, 8)


def test_load_custom_cnn_checkpoint_forwards_true_hw_to_eval_worker():
    """Fix-round-1 finding: `_load_custom_cnn_checkpoint` (the eval-on-labeled-data
    path -- `_after_load` -> `_evaluate_model_on_labeled`) used to truncate a
    non-square `ckpt["input_size"]` to `input_size[0]` before constructing the
    TorchvisionInferenceWorker, so non-square eval never happened end-to-end even
    though `_build_transform` itself was fixed. It must now forward the full
    (H, W) tuple through `_run_torchvision_inference` unchanged.
    """
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    main_window_module = pytest.importorskip("hydra_suite.classkit.gui.main_window")
    MainWindow = main_window_module.MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        captured = {}

        def _fake_run_torchvision_inference(self, model_path, class_names, **kwargs):
            captured["input_size"] = kwargs.get("input_size")

        window._run_torchvision_inference = _fake_run_torchvision_inference.__get__(
            window
        )

        input_h, input_w = 64, 128
        ckpt = {
            "arch": "resnet18",
            "class_names": ["left", "right"],
            "input_size": [input_h, input_w],
        }
        window._load_custom_cnn_checkpoint(Path("/tmp/fake_model.pth"), ckpt)

        assert captured["input_size"] == (input_h, input_w)
        assert captured["input_size"] != input_h  # not truncated to a bare int
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
