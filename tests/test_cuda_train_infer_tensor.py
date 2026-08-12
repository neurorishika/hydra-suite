"""CUDA train==infer guard for the canonical Layer 2 (letterbox) transform.

Training builds its model-input tensor via ``CanonicalFitTransform`` (CPU,
uint8 BGR in, ``apply_fit`` -> ``letterbox_fit`` under the hood -- see
``training/canonical_transform.py``). Inference on the CUDA path builds the
SAME model-input tensor by calling ``letterbox_fit`` directly on a
CUDA-resident float32 CHW crop tensor (see ``core/inference/stages/cnn.py``'s
``run_cnn_batch`` GPU branch).

This test proves those two call paths agree, for the SAME crop, on the
un-augmented (no flip/jitter/normalisation) path -- the guard that was
missing: nothing previously proved train and CUDA-infer produce the same
tensor for the same input.

CUDA-only: this box (MPS) skips it; it runs for real on mehek.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hydra_suite.training.canonical_transform import CanonicalFitTransform


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA box only")
def test_cuda_infer_equals_training_transform():
    from hydra_suite.core.canonicalization.resample import letterbox_fit

    rng = np.random.default_rng(42)
    # Deliberately non-square crop and non-square model input, so an
    # (H, W)/(W, H) mixup between the two paths would be caught.
    crop_bgr_u8 = rng.integers(0, 256, (70, 110, 3), np.uint8)
    model_hw = (64, 96)  # (H, W)
    model_wh = (model_hw[1], model_hw[0])  # (W, H)

    # --- Training path: CPU, uint8 BGR -> CanonicalFitTransform -> [0,1] CHW.
    train_fitted_u8 = CanonicalFitTransform(model_hw)(crop_bgr_u8)  # HWC uint8
    assert train_fitted_u8.shape == (model_hw[0], model_hw[1], 3)
    train_tensor = (
        torch.from_numpy(train_fitted_u8.transpose(2, 0, 1)).float().div(255.0)
    )  # (3, H, W) CPU

    # --- Inference path: CUDA, float32 CHW [0,1] -> letterbox_fit directly.
    crop_chw_cuda = (
        torch.from_numpy(crop_bgr_u8.transpose(2, 0, 1)).float().div(255.0).to("cuda")
    )
    infer_tensor_cuda = letterbox_fit(crop_chw_cuda, model_wh)  # (3, H, W) CUDA
    infer_tensor = infer_tensor_cuda.detach().cpu()

    assert infer_tensor.shape == train_tensor.shape
    # apply_fit round-trips through uint8 (rounds to the nearest 1/255) while
    # the CUDA path stays in float throughout, so allow a little more than
    # one quantization step of slack; anything larger indicates a real
    # train/infer divergence (aspect ratio, offset, or resampler mismatch).
    max_abs_diff = float((train_tensor - infer_tensor).abs().max())
    assert max_abs_diff < 4.0 / 255.0, (
        f"train (CPU CanonicalFitTransform) vs infer (CUDA letterbox_fit) "
        f"model-input tensors diverged: max|diff|={max_abs_diff}"
    )
