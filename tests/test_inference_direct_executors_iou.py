"""Configured IoU must reach non_max_suppression on the direct executors,
not the hardcoded 0.5. Uses a monkeypatched NMS / fake executor (no GPU/engine)."""

import torch

import hydra_suite.core.inference.direct_executors as de


def test_segment_decode_honors_iou_thres(monkeypatch):
    captured = {}

    def fake_nms(pred, *args, **kwargs):
        captured["iou"] = kwargs.get("iou_thres")
        return [torch.zeros((0, 6 + 32))]

    monkeypatch.setattr(
        (
            de.nms
            if hasattr(de, "nms")
            else __import__("ultralytics.utils", fromlist=["nms"]).nms
        ),
        "non_max_suppression",
        fake_nms,
        raising=True,
    )
    preds = torch.zeros((1, 32 + 4 + 1, 8))
    protos = torch.zeros((1, 32, 8, 8))
    de._decode_segment_predictions(
        preds,
        protos,
        img_tensor_shape=(1, 3, 64, 64),
        orig_shape=(64, 64),
        conf_thres=0.25,
        classes=None,
        max_det=100,
        nc=1,
        iou_thres=0.33,
    )
    assert abs(captured["iou"] - 0.33) < 1e-9


def test_adapter_forwards_iou_to_executor():
    from hydra_suite.core.inference.runtime_artifacts import DirectExecutorAdapter

    class FakeExec:
        imgsz = 640
        names = {0: "obj"}

        def predict(self, frames, *, conf_thres, classes, max_det, iou_thres=None):
            self.seen = {"iou_thres": iou_thres, "conf_thres": conf_thres}
            return []

    ex = FakeExec()
    adapter = DirectExecutorAdapter(ex)
    adapter.predict([object()], conf=0.1, iou=0.42, classes=None)
    assert abs(ex.seen["iou_thres"] - 0.42) < 1e-9
