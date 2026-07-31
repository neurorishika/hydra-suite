"""CPU unit tests for the segment-as-OBB raw-output decode (Task 7).

Exercises _decode_segment_predictions with synthetic CPU tensors shaped like
a real YOLO-segment raw head output -- no TensorRT/ONNX session, no GPU, and
no ultralytics Results/cv2 machinery is involved.
"""

from __future__ import annotations

import torch

from hydra_suite.core.inference.direct_executors import _decode_segment_predictions


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


def test_decode_segment_predictions_scales_boxes_to_nonsquare_original_frame():
    """orig_shape != (imgsz, imgsz) makes ops.scale_boxes's gain/pad math non-identity.

    The synthetic detection has letterbox-space xyxy (22, 27, 42, 37) (derived
    from the helper's cx=32, cy=32, w=20, h=10). With imgsz=64 and a
    non-square orig_shape=(32, 64), ultralytics' own gain/pad formula gives:

        gain = min(64/32, 64/64) = 1
        pad_x = round((64 - round(64*1)) / 2 - 0.1) = 0
        pad_y = round((64 - round(32*1)) / 2 - 0.1) = 16

    so the expected original-frame box is
    ((22-0)/1, (27-16)/1, (42-0)/1, (37-16)/1) = (22, 11, 42, 21).

    This would fail if ops.scale_boxes were removed (boxes would stay at
    (22, 27, 42, 37)) or called with the wrong xywh= argument (which would
    corrupt the box format instead of just translating it).
    """
    nc, nm, mh, mw, imgsz = 1, 4, 16, 16, 64
    preds = _make_synthetic_prediction(nc, nm, num_anchors=8)
    protos = torch.ones((1, nm, mh, mw), dtype=torch.float32)

    results = _decode_segment_predictions(
        preds,
        protos,
        img_tensor_shape=(1, 3, imgsz, imgsz),
        orig_shape=(32, 64),
        conf_thres=0.05,
        classes=None,
        max_det=10,
        nc=nc,
    )

    assert len(results) == 1
    r = results[0]
    assert r.orig_shape == (32, 64)
    assert r.boxes is not None and len(r.boxes.conf) == 1
    expected = torch.tensor([22.0, 11.0, 42.0, 21.0])
    torch.testing.assert_close(r.boxes.xyxy[0], expected)


def test_decode_segment_predictions_mask_localized_to_box_region():
    """A localized proto blob must survive process_mask's crop only inside the box.

    Uses the same synthetic detection as the other tests: letterbox-space
    xyxy (22, 27, 42, 37) on a 64x64 canvas. With mh=mw=16 the proto-space
    box (via process_mask's width_ratio/height_ratio = mh/mw / imgsz = 0.25)
    rounds to rows [7, 9) and cols [6, 10) (verified against
    ultralytics.utils.ops.crop_mask's own rounding).

    Prototypes are all-negative (so the decoded mask is off everywhere)
    except two single-pixel positive spikes: one inside the box's proto
    region, one clearly outside it. Only the inside spike may remain "on"
    after process_mask's crop -- this fails if the crop/coordinate mapping
    is wrong (e.g. if the wrong box or the wrong space were used).

    Masks are emitted at LETTERBOX resolution (upsample=True), so the surviving
    blob is the bilinear upsample of the inside spike: it must sit inside the
    box, and its centroid must land on that spike's letterbox-space position
    (proto row/col * imgsz/mh).
    """
    nc, nm, mh, mw, imgsz = 1, 1, 16, 16, 64
    preds = _make_synthetic_prediction(nc, nm, num_anchors=8)

    protos = torch.full((1, nm, mh, mw), -1.0, dtype=torch.float32)
    inside_rc = (8, 8)  # inside rows[7,9) x cols[6,10)
    outside_rc = (0, 0)  # well outside the box region
    protos[0, 0, inside_rc[0], inside_rc[1]] = 5.0
    protos[0, 0, outside_rc[0], outside_rc[1]] = 5.0

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
    assert r.masks is not None and r.masks.data.shape[0] == 1
    mask = r.masks.data[0]
    assert tuple(mask.shape) == (imgsz, imgsz)
    scale = imgsz / mh  # proto -> letterbox
    on_pixels = torch.nonzero(mask, as_tuple=False).float()
    assert on_pixels.shape[0] > 0
    # Every "on" pixel lies inside the detection's letterbox-space box
    # (22, 27) - (42, 37), i.e. process_mask's crop still holds...
    assert on_pixels[:, 0].min() >= 27 and on_pixels[:, 0].max() <= 37
    assert on_pixels[:, 1].min() >= 22 and on_pixels[:, 1].max() <= 42
    # ...and the blob is centered on the INSIDE spike, not the outside one.
    centroid_r = float(on_pixels[:, 0].mean())
    centroid_c = float(on_pixels[:, 1].mean())
    assert abs(centroid_r - inside_rc[0] * scale) < 4.0
    assert abs(centroid_c - inside_rc[1] * scale) < 4.0
    # The outside spike's letterbox neighbourhood stays off.
    assert (
        mask[
            outside_rc[0] : int(outside_rc[0] * scale) + 4,
            outside_rc[1] : int(outside_rc[1] * scale) + 4,
        ].sum()
        == 0
    )


def test_decode_segment_predictions_batch_of_two_uses_own_prototypes():
    """Each image in a B=2 batch must be decoded with ITS OWN prototypes.

    Both images carry an identical confident detection (same box), but their
    prototypes place a single positive spike at DIFFERENT proto-pixel
    locations (both inside the shared box's proto-space crop region,
    rows[7,9) x cols[6,10)). If the ``protos[i] if protos.shape[0] ==
    len(filtered) else protos[0]`` selection degenerated to always using
    protos[0], image 1's decoded on-pixel would incorrectly match image 0's
    spike location instead of its own.
    """
    nc, nm, mh, mw, imgsz = 1, 1, 16, 16, 64
    preds = torch.cat(
        [_make_synthetic_prediction(nc, nm, num_anchors=8) for _ in range(2)], dim=0
    )

    protos = torch.full((2, nm, mh, mw), -1.0, dtype=torch.float32)
    spike_image0 = (7, 6)
    spike_image1 = (8, 9)
    protos[0, 0, spike_image0[0], spike_image0[1]] = 5.0
    protos[1, 0, spike_image1[0], spike_image1[1]] = 5.0

    results = _decode_segment_predictions(
        preds,
        protos,
        img_tensor_shape=(2, 3, imgsz, imgsz),
        orig_shape=(imgsz, imgsz),
        conf_thres=0.05,
        classes=None,
        max_det=10,
        nc=nc,
    )

    assert len(results) == 2
    scale = imgsz / mh  # proto -> letterbox (masks are upsampled, see task I3)
    expected_spikes = [spike_image0, spike_image1]
    for i, expected_rc in enumerate(expected_spikes):
        r = results[i]
        assert r.masks is not None and r.masks.data.shape[0] == 1
        mask = r.masks.data[0]
        on_pixels = torch.nonzero(mask, as_tuple=False).float()
        assert on_pixels.shape[0] > 0, f"image {i}: expected a decoded blob"
        row = float(on_pixels[:, 0].mean())
        col = float(on_pixels[:, 1].mean())
        assert (
            abs(row - expected_rc[0] * scale) < 4.0
            and abs(col - expected_rc[1] * scale) < 4.0
        ), (
            f"image {i}: expected blob near letterbox "
            f"{(expected_rc[0] * scale, expected_rc[1] * scale)}, got "
            f"({row}, {col}) -- prototypes were not applied per-image"
        )


def test_decode_segment_predictions_masks_at_letterbox_resolution():
    """Final review IMPORTANT 3: TRT masks must come back at the SAME resolution
    as every other runtime (letterbox/imgsz), not at coarse proto resolution.

    ``roi_align`` in ``rotated_rect_from_masks`` samples FROM THE SOURCE mask,
    so proto-resolution masks genuinely lose 4x of the angle/size signal (a
    30x12 px animal becomes ~2.5x1 proto pixels) and diverge from the same
    checkpoint's output under cuda/cpu.
    """
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

    mask = results[0].masks.data
    assert tuple(mask.shape[-2:]) == (imgsz, imgsz), (
        f"masks decoded at {tuple(mask.shape[-2:])}, expected letterbox "
        f"resolution ({imgsz}, {imgsz})"
    )
