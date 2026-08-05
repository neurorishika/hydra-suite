"""``engine_params.build_roi_mask`` must reproduce the GUI's live ROI raster.

The GUI rasterizes ``roi_shapes`` into ``self._mw.roi_mask`` in
``gui/orchestrators/session.py::_generate_combined_roi_mask`` (include=255 then
exclude=0, over ``np.zeros((height, width), np.uint8)``). Task 6 routes the GUI
param path's ROI through the shared, Qt-free ``build_roi_mask``. No gate fixture
carries an ROI (all ``roi_shapes == []``), so the oracle/characterization
cannot catch a divergence -- this focused test proves the two rasterizations
are byte-identical on a synthetic ROI (one include circle + one exclude
polygon), so the switch is behaviour-preserving.
"""

import numpy as np

from hydra_suite.trackerkit.engine_params import build_roi_mask


def _session_rasterize(roi_shapes, height, width):
    """Verbatim copy of session.py::_generate_combined_roi_mask's body.

    Kept inline (not imported) so the assertion is against the exact GUI
    algorithm even though that method lives in a Qt module.
    """
    import cv2

    if not roi_shapes:
        return None
    combined_mask = np.zeros((height, width), np.uint8)
    for shape in roi_shapes:
        if shape.get("mode", "include") == "include":
            if shape["type"] == "circle":
                cx, cy, radius = shape["params"]
                cv2.circle(combined_mask, (int(cx), int(cy)), int(radius), 255, -1)
            elif shape["type"] == "polygon":
                pts = np.array(shape["params"], dtype=np.int32)
                cv2.fillPoly(combined_mask, [pts], 255)
    for shape in roi_shapes:
        if shape.get("mode", "include") == "exclude":
            if shape["type"] == "circle":
                cx, cy, radius = shape["params"]
                cv2.circle(combined_mask, (int(cx), int(cy)), int(radius), 0, -1)
            elif shape["type"] == "polygon":
                pts = np.array(shape["params"], dtype=np.int32)
                cv2.fillPoly(combined_mask, [pts], 0)
    return combined_mask


def test_build_roi_mask_matches_session_rasterization():
    width, height = 200, 160
    roi_shapes = [
        {"type": "circle", "mode": "include", "params": [100, 80, 60]},
        {
            "type": "polygon",
            "mode": "exclude",
            "params": [[90, 70], [140, 72], [130, 120], [85, 110]],
        },
    ]

    shared = build_roi_mask(roi_shapes, width, height)
    reference = _session_rasterize(roi_shapes, height, width)

    assert shared is not None and reference is not None
    assert shared.dtype == reference.dtype == np.uint8
    assert shared.shape == reference.shape == (height, width)
    assert set(np.unique(shared)).issubset({0, 255})
    assert np.array_equal(shared, reference)


def test_build_roi_mask_empty_shapes_is_none():
    assert build_roi_mask([], 100, 100) is None
    assert build_roi_mask(None, 100, 100) is None
