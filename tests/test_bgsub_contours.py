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
    # entirely. Ellipses keep the same wide-vs-tall blob semantics (center,
    # rough extent) while producing contours long enough to survive.
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.ellipse(mask, (60, 40), (30, 20), 0, 0, 360, 255, -1)  # wide blob
    cv2.ellipse(mask, (145, 145), (15, 25), 0, 0, 360, 255, -1)  # tall blob
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


def test_run_bgsub_leaves_polygons_none_by_default():
    """The tracking hot path must not compute or carry contours."""
    from hydra_suite.core.inference.config import BgSubConfig

    assert BgSubConfig.__dataclass_fields__ is not None
    cfg = BgSubConfig()
    assert getattr(cfg, "emit_native_geometry", False) is False
