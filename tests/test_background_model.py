from __future__ import annotations

import numpy as np

from tests.helpers.module_loader import load_src_module, make_cv2_stub

background_mod = load_src_module(
    "hydra_suite/core/background/model.py",
    "background_model_under_test",
    stubs={"cv2": make_cv2_stub()},
)
BackgroundModel = background_mod.BackgroundModel


def test_background_model_cpu_fallback_initialization() -> None:
    model = BackgroundModel({"ENABLE_GPU_BACKGROUND": False})
    assert model.use_gpu is False
    assert model.gpu_type is None


def test_update_adaptive_background_numba_kernel_behaviour() -> None:
    background_mod.NUMBA_AVAILABLE = False
    model = BackgroundModel(
        {
            "ENABLE_GPU_BACKGROUND": False,
            "ENABLE_ADAPTIVE_BACKGROUND": True,
            "BACKGROUND_LEARNING_RATE": 0.1,
        }
    )

    gray0 = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    gray1 = np.array([[20, 30], [40, 50]], dtype=np.uint8)

    # First call initializes backgrounds.
    assert model.update_and_get_background(gray0, roi_mask=None) is None

    out = model.update_and_get_background(gray1, roi_mask=None)
    expected = np.array([[11.0, 21.0], [31.0, 41.0]], dtype=np.float32)
    np.testing.assert_allclose(
        model.adaptive_background, expected, atol=1e-6, rtol=1e-6
    )
    assert out.shape == gray1.shape
