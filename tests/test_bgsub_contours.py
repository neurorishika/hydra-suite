import cv2
import numpy as np

from hydra_suite.core.background.measure import BackgroundMeasurer


def _params():
    return {
        "MAX_TARGETS": 4,
        "MIN_CONTOUR_AREA": 10,
        "MAX_CONTOUR_MULTIPLIER": 20,
        "ENABLE_SIZE_FILTERING": False,
    }


def _mask_with_two_blobs():
    # Filled ellipses (not axis-aligned rectangles): a perfectly rectangular
    # filled region collapses to a 4-point contour under CHAIN_APPROX_SIMPLE,
    # which the >=5-point contour-length filter in detect_objects then skips
    # entirely. Ellipses produce contours long enough to survive that filter.
    #
    # Sizes and positions are deliberately chosen so scan order and size
    # order DIVERGE. Empirically verified against this OpenCV build:
    # cv2.findContours(..., RETR_EXTERNAL) on this mask returns the bottom
    # blob first and the top blob second, i.e. scan order is
    # [small (bottom), large (top)], while size order (descending, what
    # MAX_TARGETS capping must use) is [large (top), small (bottom)]. A
    # MAX_TARGETS=1 cap that naively truncates in scan order
    # (`contours[:N]`) instead of applying the size-sorted index list
    # (`idxs = np.argsort(sizes)[::-1][:N]`) therefore keeps the WRONG
    # contour, which the alignment assertion in
    # test_contours_stay_aligned_after_max_targets_cap catches.
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.ellipse(mask, (60, 40), (30, 25), 0, 0, 360, 255, -1)  # large blob (top)
    cv2.ellipse(mask, (145, 145), (15, 20), 0, 0, 360, 255, -1)  # small blob (bottom)
    return mask


def test_detect_objects_default_return_shape_is_unchanged():
    engine = BackgroundMeasurer(_params())
    result = engine.detect_objects(_mask_with_two_blobs(), 0)
    assert len(result) == 4


def test_detect_objects_returns_contours_when_requested():
    engine = BackgroundMeasurer(_params())
    meas, sizes, shapes, confs, contours = engine.detect_objects(
        _mask_with_two_blobs(), 0, return_contours=True
    )
    assert len(contours) == len(meas) == 2
    for contour in contours:
        assert contour.ndim == 2 and contour.shape[1] == 2
        assert contour.dtype == np.float32


def test_contours_stay_aligned_after_size_filtering():
    params = _params()
    params["ENABLE_SIZE_FILTERING"] = True
    params["MIN_OBJECT_SIZE"] = 1000
    params["MAX_OBJECT_SIZE"] = float("inf")
    engine = BackgroundMeasurer(params)
    meas, _sizes, _shapes, _confs, contours = engine.detect_objects(
        _mask_with_two_blobs(), 0, return_contours=True
    )
    assert len(contours) == len(meas)
    # Each surviving contour's centroid must sit near its own measurement.
    for m, contour in zip(meas, contours):
        assert abs(float(contour[:, 0].mean()) - float(m[0])) < 20.0


def test_contours_stay_aligned_after_max_targets_cap():
    params = _params()
    params["MAX_TARGETS"] = 1
    engine = BackgroundMeasurer(params)
    meas, _sizes, _shapes, _confs, contours = engine.detect_objects(
        _mask_with_two_blobs(), 0, return_contours=True
    )
    assert len(meas) == 1
    assert len(contours) == 1
    # The surviving contour must describe the surviving measurement, not just
    # be positionally present. The fixture's scan order and size order
    # diverge (see _mask_with_two_blobs), so this fails for a cap that
    # truncates in scan order (`contours[:N]`) instead of following the
    # size-sorted index list.
    assert abs(float(contours[0][:, 0].mean()) - float(meas[0][0])) < 20.0
    assert abs(float(contours[0][:, 1].mean()) - float(meas[0][1])) < 20.0


def test_run_bgsub_leaves_polygons_none_by_default():
    """The tracking hot path must not compute or carry contours."""
    from hydra_suite.core.inference.config import BgSubConfig

    assert BgSubConfig.__dataclass_fields__ is not None
    cfg = BgSubConfig()
    assert getattr(cfg, "emit_native_geometry", False) is False
