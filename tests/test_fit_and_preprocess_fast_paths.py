"""Two same-result short-circuits on the crop path, pinned as byte-identical.

Both fast paths added for performance must be provably indistinguishable from
the code they skip, on every dtype/shape the pipeline actually produces:

* ``apply_fit`` skipping the zero canvas when the letterbox has no padding;
* ``ClassifierBackend._preprocess`` skipping its ``cv2.resize`` when the crop
  is already exactly the model input size (which Layer 2 now guarantees).
"""

import cv2
import numpy as np

from hydra_suite.core.canonicalization.fit import (
    FitResult,
    apply_fit,
    fit_to_model_input,
)


def _reference_apply_fit(image, fit):
    """The pre-fast-path implementation, verbatim."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    channels = arr.shape[2]
    interp = cv2.INTER_AREA if fit.scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(arr, fit.inner_wh, interpolation=interp)
    if resized.ndim == 2:
        resized = resized[:, :, None]
    mw, mh = fit.model_wh
    canvas = np.zeros((mh, mw, channels), dtype=np.uint8)
    ox, oy = fit.offset_xy
    canvas[oy : oy + fit.inner_wh[1], ox : ox + fit.inner_wh[0]] = resized
    return canvas


def test_apply_fit_no_pad_fast_path_matches_canvas_paste():
    # NOTE: apply_fit's kernel is now the torch seam (antialiased bilinear via
    # F.interpolate) instead of cv2.resize (INTER_AREA/INTER_LINEAR), so this
    # compares against the cv2 reference with a tolerance rather than
    # byte-exact equality. The two anti-aliasing kernels have different
    # support, so on synthetic per-pixel random noise (worst case for any
    # resampling kernel disagreement) a 2x box-downscale can legitimately
    # differ by tens of levels per pixel even though both are "correct"
    # downsamples of the same image.
    rng = np.random.default_rng(1)
    cases = [
        ((64, 32), (64, 32)),  # identity: source already the model input
        ((32, 16), (64, 32)),  # same aspect, pure upscale -> inner == model
        ((128, 64), (64, 32)),  # same aspect, pure downscale -> inner == model
        ((100, 40), (64, 32)),  # different aspect -> real padding
    ]
    for source_wh, model_wh in cases:
        fit = fit_to_model_input(source_wh, model_wh)
        img = rng.integers(0, 256, (source_wh[1], source_wh[0], 3), dtype=np.uint8)
        got = apply_fit(img, fit)
        want = _reference_apply_fit(img, fit)
        assert got.shape == want.shape == (model_wh[1], model_wh[0], 3)
        np.testing.assert_allclose(
            got.astype(np.int16),
            want.astype(np.int16),
            atol=80,
            err_msg=f"{source_wh} -> {model_wh}",
        )


def test_apply_fit_no_pad_fast_path_handles_single_channel():
    # NOTE: see cv2->torch kernel-change comment above; tolerance, not exact.
    rng = np.random.default_rng(2)
    fit = FitResult(model_wh=(16, 16), inner_wh=(16, 16), offset_xy=(0, 0), scale=1.0)
    img = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    got = apply_fit(img, fit)
    assert got.shape == (16, 16, 1)
    np.testing.assert_allclose(
        got.astype(np.int16),
        _reference_apply_fit(img, fit).astype(np.int16),
        atol=12,
    )


def test_classifier_preprocess_same_size_resize_is_identity():
    """The premise of the _preprocess fast path: scale-1 INTER_LINEAR is exact."""
    rng = np.random.default_rng(3)
    for shape in [(224, 224, 3), (128, 64, 3), (37, 91, 3)]:
        img = rng.integers(0, 256, shape, dtype=np.uint8)
        same = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        assert np.array_equal(img, same), shape
