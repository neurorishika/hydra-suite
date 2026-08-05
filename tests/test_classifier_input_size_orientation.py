"""input_size is (H, W) everywhere. Non-square models make this observable."""

import numpy as np

from hydra_suite.core.identity.classification.backend import _normalize_input_size


def test_normalize_returns_h_w():
    assert _normalize_input_size([64, 128]) == (64, 128)


def test_model_fit_honours_h_w_order():
    """A crop fitted to a non-square ``input_size`` lands at exactly (H, W).

    This used to assert the shape of ``extract_classifier_crops`` directly,
    which warped straight to the model input.  Under global canonicalization
    that function returns a CANONICAL crop and the model-input shape is Layer
    2's job, so the (H, W) contract is asserted where it now lives.  Getting
    the order wrong here transposes every non-square classifier input, and
    tiny head-tail models default to a non-square [64, 128].
    """
    from hydra_suite.core.canonicalization.fit import apply_fit, fit_to_model_input

    in_h, in_w = (64, 128)  # (H, W) -- deliberately non-square
    crop = np.zeros((26, 60, 3), dtype=np.uint8)  # a canonical crop (H, W, C)

    fit = fit_to_model_input((crop.shape[1], crop.shape[0]), (in_w, in_h))
    out = apply_fit(crop, fit)

    assert out.shape[:2] == (in_h, in_w)


def test_custom_cnn_params_accept_a_pair():
    from hydra_suite.training.contracts import CustomCNNParams

    p = CustomCNNParams(input_size=(64, 128))
    assert tuple(p.input_size) == (64, 128)
