"""The optimizer cache builder must drive InferenceRunner.run_batch_pass, not
a legacy YOLOOBBDetector."""

import types

import hydra_suite.trackerkit.gui.workers.param_optimizer_worker as ow


def test_cache_build_worker_uses_run_batch_pass(monkeypatch, tmp_path):
    calls = {}

    class _FakeRunner:
        def __init__(
            self, cfg, cache_dir=None, video_path=None, cache_only=False, roi_mask=None
        ):
            calls["cache_dir"] = cache_dir
            calls["roi_mask"] = roi_mask

        def run_batch_pass(
            self,
            video_path,
            progress_cb=None,
            start_frame=0,
            end_frame=None,
            should_stop=None,
        ):
            calls["ran"] = True
            if progress_cb is not None:
                progress_cb(1, 1)

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(ow, "InferenceRunner", _FakeRunner, raising=False)
    monkeypatch.setattr(
        ow, "inference_config_for_optimizer_params", lambda p: object(), raising=False
    )

    emitted = []
    worker = ow.DetectionCacheBuildWorker(
        video_path="v.mp4",
        cache_dir=str(tmp_path),
        params={"ROI_MASK": "roi-mask"},
        start_frame=0,
        end_frame=1,
    )
    worker.finished_signal = types.SimpleNamespace(emit=lambda *a: emitted.append(a))
    worker.progress_signal = types.SimpleNamespace(emit=lambda *a: None)
    worker.run()
    assert calls.get("ran") is True
    assert calls.get("closed") is True
    assert calls["roi_mask"] == "roi-mask"
    assert emitted and emitted[-1][0] is True
    assert emitted[-1][1] == str(tmp_path)
