import sys

import numpy as np
import pytest

from hydra_suite.core.inference.sam2 import executor as ex


def test_resolve_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(ex, "TORCH_CUDA_AVAILABLE", True)
    monkeypatch.setattr(ex, "MPS_AVAILABLE", False)
    assert ex.resolve_sam2_device() == "cuda"


def test_resolve_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(ex, "TORCH_CUDA_AVAILABLE", False)
    monkeypatch.setattr(ex, "MPS_AVAILABLE", False)
    assert ex.resolve_sam2_device() == "cpu"


def test_segment_picks_highest_iou_mask():
    # Inject a fake SAM2 image-predictor: predict() returns 3 masks + 3 ious.
    class _FakePredictor:
        def set_image(self, rgb):
            self.rgb = rgb

        def predict(
            self, box=None, point_coords=None, point_labels=None, multimask_output=True
        ):
            masks = np.stack(
                [
                    np.zeros((4, 4), bool),
                    np.ones((4, 4), bool),  # best
                    np.zeros((4, 4), bool),
                ]
            )
            ious = np.array([0.1, 0.9, 0.2])
            return masks, ious, None

    e = ex.Sam2SegmentExecutor(_FakePredictor())
    e.set_image(np.zeros((4, 4, 3), np.uint8))
    mask, iou = e.segment((0, 0, 4, 4), [(2, 2)], [(0, 0)])
    assert iou == 0.9 and mask.all()


@pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="torch device")
def test_real_sam2_segment_smoke():
    pytest.importorskip("sam2")
    from hydra_suite.core.inference.sam2.checkpoints import DEFAULT_VARIANT
    from hydra_suite.core.inference.sam2.executor import Sam2SegmentExecutor

    try:
        ex_instance = Sam2SegmentExecutor.from_variant(DEFAULT_VARIANT)
    except Exception as e:  # weights not downloaded in CI, etc.
        pytest.skip(f"SAM2 weights unavailable: {e}")
    img = (np.random.rand(256, 256, 3) * 255).astype("uint8")
    ex_instance.set_image(img)
    mask, iou = ex_instance.segment((50, 50, 200, 200), [(125, 125)], [])
    assert mask.shape == (256, 256) and 0.0 <= iou <= 1.0
