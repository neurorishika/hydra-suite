"""detect_batch returns filtered OBBResults in memory, mirroring run_realtime's
detect+filter prefix, without touching any cache."""

import numpy as np

from hydra_suite.core.inference.runner import InferenceRunner


class _FakeOBBModels:
    def close(self):
        pass


def _make_runner_with_fakes(monkeypatch, per_frame_counts):
    # Bypass __init__ so no real models load.
    runner = InferenceRunner.__new__(InferenceRunner)
    runner.config = _make_obb_only_config()
    runner._models = type("M", (), {"obb": _FakeOBBModels(), "bgsub": None})()
    runner.runtime = object()
    runner._caches = None

    import hydra_suite.core.inference.runner as rmod
    from hydra_suite.core.inference.result import OBBResult

    def fake_run_obb(frames, models, cfg, runtime):
        out = []
        for n in per_frame_counts[: len(frames)]:
            out.append(
                OBBResult(
                    frame_idx=0,
                    centroids=np.zeros((n, 2), np.float32),
                    angles=np.zeros(n, np.float32),
                    sizes=np.ones(n, np.float32),
                    shapes=np.ones((n, 2), np.float32),
                    confidences=np.ones(n, np.float32),
                    corners=np.zeros((n, 4, 2), np.float32),
                    detection_ids=OBBResult.make_detection_ids(0, n),
                )
            )
        return out

    def fake_filter_for_source(config, raw_obb, roi_mask=None):
        return raw_obb, np.arange(raw_obb.num_detections, dtype=np.int32)

    monkeypatch.setattr(rmod, "run_obb", fake_run_obb)
    monkeypatch.setattr(rmod, "filter_for_source", fake_filter_for_source)
    return runner


def _make_obb_only_config():
    from hydra_suite.core.inference.config import build_obb_only_config

    return build_obb_only_config("m.pt", compute_runtime="cpu")


def test_detect_batch_returns_one_result_per_frame_with_frame_idx(monkeypatch):
    runner = _make_runner_with_fakes(monkeypatch, per_frame_counts=[3, 0, 5])
    frames = [np.zeros((8, 8, 3), np.uint8) for _ in range(3)]
    results = runner.detect_batch(frames, frame_indices=[10, 11, 12])
    assert [r.num_detections for r in results] == [3, 0, 5]
    assert [r.frame_idx for r in results] == [10, 11, 12]
