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
