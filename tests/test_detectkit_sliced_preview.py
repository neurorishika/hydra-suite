import numpy as np

from hydra_suite.core.inference.result import OBBResult
from hydra_suite.detectkit.gui import prediction_preview as pp


class _FakeResult:
    """Minimal stand-in ultralytics result for one tile with zero detections."""

    def __init__(self):
        self.obb = None
        self.boxes = None


class _FakeExecutor:
    """Records the tile batch it was asked to predict; returns empty results."""

    def __init__(self):
        self.calls = []

    def predict(self, images, **kw):
        self.calls.append([np.asarray(im).shape for im in images])
        return [_FakeResult() for _ in images]


class _MPSFakeExecutor(_FakeExecutor):
    """Executor shape sufficient to exercise the MPS tile-batch guard."""

    class _Model:
        class _Parameter:
            device = "mps:0"

        def parameters(self):
            return iter([self._Parameter()])

    def __init__(self):
        super().__init__()
        self.model = self._Model()


def test_predict_sliced_tiles_and_merges_empty(monkeypatch):
    # Force extract_obb_result to yield empty OBBResults so we exercise tiling+merge
    # without a real model.
    monkeypatch.setattr(
        pp,
        "extract_obb_result",
        lambda res, frame_idx=0, **kw: OBBResult(
            frame_idx=frame_idx,
            centroids=np.zeros((0, 2), np.float32),
            angles=np.zeros((0,), np.float32),
            sizes=np.zeros((0,), np.float32),
            shapes=np.zeros((0, 2), np.float32),
            confidences=np.zeros((0,), np.float32),
            corners=np.zeros((0, 4, 2), np.float32),
            detection_ids=np.zeros((0,), np.int64),
        ),
    )
    frame = np.zeros((512, 512, 3), np.uint8)
    ex = _FakeExecutor()
    out = pp.predict_sliced_obb_result(
        ex,
        frame,
        geometry_mode="custom",
        imgsz=640,
        reference_body_px=0.0,
        object_tile_fraction=0.15,
        slice_width=256,
        slice_height=256,
        overlap=0.2,
        merge_threshold=0.5,
        confidence_threshold=0.25,
    )
    assert out is not None
    assert out.num_detections == 0
    # 512x512 with 256 tiles + 0.2 overlap tiles into a >1 tile grid.
    assert len(ex.calls[0]) > 1


def test_predict_sliced_bounds_mps_tile_batches(monkeypatch):
    monkeypatch.setattr(
        pp,
        "extract_obb_result",
        lambda res, frame_idx=0, **kw: OBBResult(
            frame_idx=frame_idx,
            centroids=np.zeros((0, 2), np.float32),
            angles=np.zeros((0,), np.float32),
            sizes=np.zeros((0,), np.float32),
            shapes=np.zeros((0, 2), np.float32),
            confidences=np.zeros((0,), np.float32),
            corners=np.zeros((0, 4, 2), np.float32),
            detection_ids=np.zeros((0,), np.int64),
        ),
    )
    ex = _MPSFakeExecutor()

    pp.predict_sliced_obb_result(
        ex,
        np.zeros((1024, 1024, 3), np.uint8),
        geometry_mode="custom",
        imgsz=640,
        reference_body_px=0.0,
        object_tile_fraction=0.15,
        slice_width=128,
        slice_height=128,
        overlap=0.0,
        merge_threshold=0.5,
        confidence_threshold=0.25,
    )

    assert len(ex.calls) > 1
    assert max(len(batch) for batch in ex.calls) <= pp._MPS_SLICE_BATCH_SIZE


def test_predict_sliced_stops_between_tile_batches(monkeypatch):
    cancelled = False

    class _CancellingExecutor(_MPSFakeExecutor):
        def predict(self, images, **kw):
            nonlocal cancelled
            results = super().predict(images, **kw)
            cancelled = True
            return results

    ex = _CancellingExecutor()
    out = pp.predict_sliced_obb_result(
        ex,
        np.zeros((1024, 1024, 3), np.uint8),
        geometry_mode="custom",
        imgsz=640,
        reference_body_px=0.0,
        object_tile_fraction=0.15,
        slice_width=128,
        slice_height=128,
        overlap=0.0,
        merge_threshold=0.5,
        confidence_threshold=0.25,
        should_stop=lambda: cancelled,
    )

    assert out is None
    assert len(ex.calls) == 1


def test_predict_sliced_offsets_detection_into_frame_space(monkeypatch):
    # A single detection in tile (256,256)-(512,512) must land near frame (300,300).
    # The preview now applies the tile->frame offset via extract_obb_result's own
    # ``offset=`` argument (the retired ``_offset_result`` did this before), so the
    # mock must honor ``offset`` exactly as the real extractor does.
    def _fake_extract(res, frame_idx=0, offset=(0.0, 0.0), **kw):
        ox, oy = float(offset[0]), float(offset[1])
        return (
            OBBResult(
                frame_idx=frame_idx,
                centroids=np.array(
                    [[44.0 + ox, 44.0 + oy]], np.float32
                ),  # tile-local + offset
                angles=np.array([0.0], np.float32),
                sizes=np.array([100.0], np.float32),
                shapes=np.array([[100.0, 1.0]], np.float32),
                confidences=np.array([0.9], np.float32),
                corners=np.array(
                    [
                        [
                            [39 + ox, 39 + oy],
                            [49 + ox, 39 + oy],
                            [49 + ox, 49 + oy],
                            [39 + ox, 49 + oy],
                        ]
                    ],
                    np.float32,
                ),
                detection_ids=np.array([0], np.int64),
            )
            if getattr(res, "tag", "") == "hit"
            else OBBResult(
                frame_idx=frame_idx,
                centroids=np.zeros((0, 2), np.float32),
                angles=np.zeros((0,), np.float32),
                sizes=np.zeros((0,), np.float32),
                shapes=np.zeros((0, 2), np.float32),
                confidences=np.zeros((0,), np.float32),
                corners=np.zeros((0, 4, 2), np.float32),
                detection_ids=np.zeros((0,), np.int64),
            )
        )

    monkeypatch.setattr(pp, "extract_obb_result", _fake_extract)

    class _Exec:
        def predict(self, images, **kw):
            out = []
            for i, _ in enumerate(images):
                r = _FakeResult()
                r.tag = "hit" if i == len(images) - 1 else ""
                out.append(r)
            return out

    frame = np.zeros((512, 512, 3), np.uint8)
    out = pp.predict_sliced_obb_result(
        _Exec(),
        frame,
        geometry_mode="custom",
        imgsz=640,
        reference_body_px=0.0,
        object_tile_fraction=0.15,
        slice_width=256,
        slice_height=256,
        overlap=0.0,
        merge_threshold=0.5,
        confidence_threshold=0.25,
    )
    assert out.num_detections == 1
    # last tile starts at (256,256); local (44,44) -> frame ~ (300,300).
    assert abs(out.centroids[0][0] - 300.0) < 2.0
    assert abs(out.centroids[0][1] - 300.0) < 2.0
