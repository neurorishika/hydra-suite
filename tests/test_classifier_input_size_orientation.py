"""input_size is (H, W) everywhere. Non-square models make this observable."""

import numpy as np

from hydra_suite.core.identity.classification.backend import _normalize_input_size


def test_normalize_returns_h_w():
    assert _normalize_input_size([64, 128]) == (64, 128)


def test_crop_target_uses_width_second():
    """extract_classifier_crops must produce (H, W) == metadata.input_size."""
    from hydra_suite.core.inference.result import OBBResult
    from hydra_suite.core.inference.stages.crops import extract_classifier_crops

    corners = np.array(
        [[10.0, 10.0], [42.0, 10.0], [42.0, 26.0], [10.0, 26.0]], dtype=np.float32
    )
    obb = OBBResult(
        frame_idx=0,
        centroids=np.array([[26.0, 18.0]], dtype=np.float32),
        angles=np.array([0.0], dtype=np.float32),
        sizes=np.array([512.0], dtype=np.float32),
        shapes=np.array([[512.0, 2.0]], dtype=np.float32),
        confidences=np.array([0.9], dtype=np.float32),
        corners=np.stack([corners]),
        detection_ids=np.array([0], dtype=np.int64),
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    input_size = (64, 128)  # (H, W) -- deliberately non-square
    crops = extract_classifier_crops(frame, obb, input_size, 2.0, 1.3)
    assert crops[0].shape[:2] == input_size


def test_custom_cnn_params_accept_a_pair():
    from hydra_suite.training.contracts import CustomCNNParams

    p = CustomCNNParams(input_size=(64, 128))
    assert tuple(p.input_size) == (64, 128)
