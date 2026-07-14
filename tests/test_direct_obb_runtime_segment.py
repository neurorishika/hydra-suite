"""CPU unit tests for the segment-as-OBB raw-output decode (Task 7).

Exercises _decode_segment_predictions with synthetic CPU tensors shaped like
a real YOLO-segment raw head output -- no TensorRT/ONNX session, no GPU, and
no ultralytics Results/cv2 machinery is involved.
"""

from __future__ import annotations

import torch

from hydra_suite.core.detectors._direct_obb_runtime import _decode_segment_predictions


def _make_synthetic_prediction(nc: int, nm: int, num_anchors: int) -> torch.Tensor:
    """Build a (1, 4+nc+nm, num_anchors) raw-head tensor with one confident box."""
    pred = torch.zeros((1, 4 + nc + nm, num_anchors), dtype=torch.float32)
    pred[0, 0, 0] = 32.0  # cx
    pred[0, 1, 0] = 32.0  # cy
    pred[0, 2, 0] = 20.0  # w
    pred[0, 3, 0] = 10.0  # h
    pred[0, 4, 0] = 5.0  # class-0 score
    pred[0, 4 + nc : 4 + nc + nm, 0] = 1.0  # mask coefficients
    return pred


def test_decode_segment_predictions_returns_duck_typed_detections():
    nc, nm, mh, mw, imgsz = 1, 4, 16, 16, 64
    preds = _make_synthetic_prediction(nc, nm, num_anchors=8)
    protos = torch.ones((1, nm, mh, mw), dtype=torch.float32)

    results = _decode_segment_predictions(
        preds,
        protos,
        img_tensor_shape=(1, 3, imgsz, imgsz),
        orig_shape=(imgsz, imgsz),
        conf_thres=0.05,
        classes=None,
        max_det=10,
        nc=nc,
    )

    assert len(results) == 1
    r = results[0]
    assert r.orig_shape == (imgsz, imgsz)
    assert r.boxes is not None and len(r.boxes.conf) == 1
    assert r.masks is not None and r.masks.data.shape[0] == 1
    # Uniform-positive prototypes + all-ones coefficients -> a non-degenerate
    # (all-"on") decoded mask.
    assert r.masks.data.sum() > 0


def test_decode_segment_predictions_empty_below_threshold():
    nc, nm, mh, mw, imgsz = 1, 4, 8, 8, 32
    preds = torch.zeros((1, 4 + nc + nm, 4), dtype=torch.float32)
    protos = torch.zeros((1, nm, mh, mw), dtype=torch.float32)

    results = _decode_segment_predictions(
        preds,
        protos,
        img_tensor_shape=(1, 3, imgsz, imgsz),
        orig_shape=(imgsz, imgsz),
        conf_thres=0.5,
        classes=None,
        max_det=10,
        nc=nc,
    )
    assert len(results) == 1
    assert results[0].boxes is None or len(results[0].boxes.conf) == 0
