"""Lightweight post-hoc filter for cached raw YOLO detections."""

from ._obb_geometry import OBBGeometryMixin


class DetectionFilter(OBBGeometryMixin):
    """
    Lightweight post-hoc filter for cached raw YOLO detections.

    Contains only confidence thresholding and OBB IOU NMS — the exact same logic
    used by YOLOOBBDetector.filter_raw_detections — with no model loading.  Safe
    to instantiate cheaply inside inner optimizer loops.

    Usage::

        filt = DetectionFilter(params)
        meas, sizes, shapes, confs, corners, *_ = filt.filter_raw_detections(
            raw_meas, raw_sizes, raw_shapes, raw_confidences, raw_obb_corners
        )
    """

    def __init__(self, params):
        self.params = params
