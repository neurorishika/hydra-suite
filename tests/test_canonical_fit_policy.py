import numpy as np
import pytest
import torch

from hydra_suite.core.canonicalization.fit import (
    apply_fit,
    fit_crops_for_model,
    fit_to_model_input,
)
from hydra_suite.core.canonicalization.resample import (
    fit_batch_for_model,
    letterbox_fit,
    squash_fit,
)


def _crop(h=66, w=148, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_letterbox_policy_is_byte_identical_to_apply_fit():
    c = _crop()
    ref = apply_fit(c, fit_to_model_input((148, 66), (128, 128)))
    out = fit_crops_for_model([c], (128, 128), "letterbox")[0]
    assert np.array_equal(out, ref)


def test_squash_policy_fills_canvas_no_black_bars():
    c = _crop()
    c[:] = 200
    out = fit_crops_for_model([c], (128, 128), "squash")[0]
    assert out.shape == (128, 128, 3)
    assert out.min() >= 195  # no zero rows anywhere


def test_squash_matches_torch_antialiased_bilinear():
    c = _crop()
    chw = torch.from_numpy(c).permute(2, 0, 1).float()
    ref = (
        torch.nn.functional.interpolate(
            chw[None],
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0]
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    out = fit_crops_for_model([c], (128, 128), "squash")[0]
    assert np.array_equal(out, ref)


def test_torch_batch_policy_dispatch():
    x = torch.rand(4, 3, 66, 148)
    assert torch.equal(
        fit_batch_for_model(x, (128, 128), "letterbox"), letterbox_fit(x, (128, 128))
    )
    assert fit_batch_for_model(x, (128, 128), "squash").shape == (4, 3, 128, 128)
    assert torch.equal(
        fit_batch_for_model(x, (128, 128), "squash"), squash_fit(x, (128, 128))
    )


def test_torch_batch_native_policy_is_noop():
    # H, W (66, 148) deliberately differ from model_wh (128, 128) so the
    # assertion proves this is a true no-op, not an accidental shape match.
    x = torch.rand(4, 3, 66, 148)
    out = fit_batch_for_model(x, (128, 128), "native")
    assert out.shape == x.shape
    assert torch.equal(out, x)


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        fit_crops_for_model([_crop()], (128, 128), "stretchy")
