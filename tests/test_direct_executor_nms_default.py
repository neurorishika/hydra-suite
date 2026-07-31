"""Regression guard: direct-executor raw-CBC NMS must keep its protective
iou_thres=0.5 default, driven the way production actually calls it.

Context (see 9cebed6a and its revert): a prior commit threaded a caller-
supplied ``iou_thres`` through ``_BaseDirectOBBExecutor._postprocess`` and
made ``DirectExecutorAdapter.predict`` forward ``iou_thres=iou`` into it.
``stages/obb.py`` always calls ``model.predict(..., iou=1.0, ...)``
deliberately -- the real NMS at ``config.iou_threshold`` is applied later,
executor-agnostically, in ``stages/filtering.py``. Forwarding iou=1.0 into
the executor's own NMS defeated its PROTECTIVE pre-dedup for raw-CBC
(non-end2end) TensorRT/ONNX engines: with no suppression, duplicate anchors
saturate the small ``max_det`` slots and distinct animals get dropped.

This test pins the executor's own iou_thres at 0.5 for raw-CBC
(``_end2end=False``) executors, regardless of what the caller passes in,
by driving the code the way production does: build a real
``DirectExecutorAdapter`` around a raw-CBC ``_BaseDirectOBBExecutor``
instance and call ``.predict(..., iou=1.0, ...)``, then assert the
underlying ``ultralytics.utils.nms.non_max_suppression`` actually received
``iou_thres=0.5``.
"""

from __future__ import annotations

import torch

from hydra_suite.core.inference import direct_executors as de
from hydra_suite.core.inference import runtime_artifacts as ra


def _make_raw_cbc_executor() -> de._BaseDirectOBBExecutor:
    """Build a minimal raw-CBC (_end2end=False) executor instance.

    Bypasses ``__init__`` (which needs a real artifact file + ultralytics/
    torch setup) and sets only the attributes ``_postprocess`` reads:
    ``names``, ``nc``, and ``_end2end``.
    """
    executor = object.__new__(de._BaseDirectOBBExecutor)
    executor.names = {0: "animal"}
    executor.nc = 1
    executor._end2end = False
    return executor


def _make_synthetic_raw_preds(nc: int, num_anchors: int) -> torch.Tensor:
    """(1, 4+nc+1, num_anchors) raw OBB head output with one confident box.

    Layout matches _BaseDirectOBBExecutor._postprocess's expectation:
    [cx, cy, w, h, cls_scores..., angle].
    """
    preds = torch.zeros((1, 4 + nc + 1, num_anchors), dtype=torch.float32)
    preds[0, 0, 0] = 32.0  # cx
    preds[0, 1, 0] = 32.0  # cy
    preds[0, 2, 0] = 20.0  # w
    preds[0, 3, 0] = 10.0  # h
    preds[0, 4, 0] = 5.0  # class-0 score
    preds[0, 4 + nc, 0] = 0.0  # angle
    return preds


def test_raw_cbc_executor_postprocess_uses_protective_iou_default(monkeypatch):
    """Driving _postprocess directly: raw-CBC executors must NMS at 0.5."""
    captured = {}
    from ultralytics.utils import nms as ultra_nms

    real_nms = ultra_nms.non_max_suppression

    def _spy_nms(*args, **kwargs):
        captured["iou_thres"] = kwargs.get("iou_thres")
        return real_nms(*args, **kwargs)

    monkeypatch.setattr(ultra_nms, "non_max_suppression", _spy_nms)

    executor = _make_raw_cbc_executor()
    img_tensor = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    orig_frame = torch.zeros((64, 64, 3), dtype=torch.uint8).numpy()
    raw_preds = _make_synthetic_raw_preds(nc=1, num_anchors=8)

    executor._postprocess(
        raw_preds,
        img_tensor,
        [orig_frame],
        conf_thres=0.05,
        classes=None,
        max_det=8,
    )

    assert captured["iou_thres"] == 0.5


def test_adapter_predict_iou_1_0_does_not_relax_protective_nms(monkeypatch):
    """End-to-end through DirectExecutorAdapter.predict(iou=1.0), mirroring
    exactly how stages/obb.py._run_direct calls it: the caller's iou=1.0
    must NOT reach the executor's NMS -- it must still fire at 0.5."""
    captured = {}
    from ultralytics.utils import nms as ultra_nms

    real_nms = ultra_nms.non_max_suppression

    def _spy_nms(*args, **kwargs):
        captured["iou_thres"] = kwargs.get("iou_thres")
        return real_nms(*args, **kwargs)

    monkeypatch.setattr(ultra_nms, "non_max_suppression", _spy_nms)

    # _predict_chunk / predict need _model_batch_size / _static_batch lookups
    # via getattr with defaults, and _preprocess for the numpy-frame path --
    # avoid all of that by calling _postprocess directly through a thin
    # stand-in "predict" that matches the executor's real predict() shape
    # for a single, already-preprocessed chunk.
    img_tensor = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    orig_frame = torch.zeros((64, 64, 3), dtype=torch.uint8).numpy()
    raw_preds = _make_synthetic_raw_preds(nc=1, num_anchors=8)

    class _FixedInferenceExecutor(de._BaseDirectOBBExecutor):
        def _run_inference(self, img_tensor):
            return raw_preds

        def _preprocess(self, frames):
            return img_tensor

    fixed_executor = object.__new__(_FixedInferenceExecutor)
    fixed_executor.names = {0: "animal"}
    fixed_executor.nc = 1
    fixed_executor._end2end = False

    adapter = ra.DirectExecutorAdapter(fixed_executor, max_det=8)
    adapter.predict([orig_frame], conf=0.05, iou=1.0, classes=None)

    assert captured["iou_thres"] == 0.5
