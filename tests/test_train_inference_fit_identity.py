"""The same image, fitted by training and by inference, is byte-identical."""

import numpy as np
import pytest

from hydra_suite.training.canonical_transform import CanonicalFitTransform


@pytest.fixture
def crop():
    rng = np.random.default_rng(1234)
    return rng.integers(0, 255, (64, 128, 3), dtype=np.uint8)


@pytest.mark.parametrize("model_hw", [(224, 224), (64, 128), (96, 160)])
def test_classkit_train_matches_inference(crop, model_hw):
    from hydra_suite.core.canonicalization.fit import apply_fit, fit_to_model_input

    train_out = CanonicalFitTransform(model_hw)(crop)
    fit = fit_to_model_input((crop.shape[1], crop.shape[0]), (model_hw[1], model_hw[0]))
    infer_out = apply_fit(crop, fit)
    np.testing.assert_array_equal(np.asarray(train_out), infer_out)


@pytest.mark.parametrize("model_hw", [(256, 192), (256, 256)])
def test_posekit_train_matches_inference(crop, model_hw):
    from hydra_suite.core.canonicalization.fit import apply_fit, fit_to_model_input

    train_out = CanonicalFitTransform(model_hw)(crop)
    fit = fit_to_model_input((crop.shape[1], crop.shape[0]), (model_hw[1], model_hw[0]))
    np.testing.assert_array_equal(np.asarray(train_out), apply_fit(crop, fit))


def test_transform_rejects_float_input(crop):
    with pytest.raises(TypeError):
        CanonicalFitTransform((224, 224))(crop.astype(np.float32) / 255.0)


def test_no_resize_call_survives_in_the_runner():
    from pathlib import Path

    runner = Path(__file__).resolve().parents[1] / "src/hydra_suite/training/runner.py"
    assert "Resize((sz, sz))" not in runner.read_text(encoding="utf-8")


def test_tiny_train_matches_tiny_inference_nonsquare(tmp_path):
    """The tiny-classifier training dataset and ClassKit tiny-inference worker
    must fit a non-square source into a non-square model input identically,
    AND both must match the Layer 2 letterbox reference (not an anisotropic
    stretch).

    Uses a source aspect (200x120, 5:3) that differs from the model input
    aspect (128x64, 2:1), so an anisotropic ``cv2.resize`` to (128, 64) would
    produce different pixels than an isotropic letterbox -- a square-only
    test (train==infer, both wrong the same way) cannot distinguish "both
    letterboxed" from "both anisotropically stretched", which is exactly how
    the original bug passed the pre-existing (square-only) test suite.
    """
    import cv2
    import numpy as np

    from hydra_suite.classkit.jobs.task_workers import TinyCNNInferenceWorker
    from hydra_suite.core.canonicalization.fit import apply_fit, fit_to_model_input
    from hydra_suite.training.runner import _build_tiny_dataset_class

    rng = np.random.default_rng(99)
    bgr = rng.integers(0, 255, (120, 200, 3), dtype=np.uint8)
    path = tmp_path / "crop.png"
    cv2.imwrite(str(path), bgr)

    input_w, input_h = 128, 64  # non-square model input (H, W) = (64, 128)

    TinyDataset = _build_tiny_dataset_class(input_w, input_h)
    train_ds = TinyDataset([(path, 0)], augment=False)
    train_x, _ = train_ds[0]
    train_arr = (train_x.numpy() * 255.0).round().astype(np.uint8)

    infer_batch = TinyCNNInferenceWorker._load_batch_images(
        [path], input_w, input_h, force_monochrome=False
    )
    infer_arr = (infer_batch[0] * 255.0).round().astype(np.uint8)

    np.testing.assert_array_equal(train_arr, infer_arr)

    # Reference: read the same file the way both workers do (BGR->RGB), then
    # letterbox via Layer 2 directly. This is the ground truth the bug (an
    # anisotropic cv2.resize) diverges from.
    rgb = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    fit = fit_to_model_input((rgb.shape[1], rgb.shape[0]), (input_w, input_h))
    reference = apply_fit(rgb, fit)
    reference_chw = reference.transpose(2, 0, 1)

    np.testing.assert_array_equal(train_arr, reference_chw)
    np.testing.assert_array_equal(infer_arr, reference_chw)
