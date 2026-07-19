"""The optimizer cache builder must drive InferenceRunner.run_batch_pass, not
a legacy YOLOOBBDetector."""

import types

import hydra_suite.core.tracking.optimization.optimizer_workers as ow


def test_cache_build_worker_uses_run_batch_pass(monkeypatch, tmp_path):
    calls = {}

    class _FakeRunner:
        def __init__(self, cfg, cache_dir=None, video_path=None, cache_only=False):
            calls["cache_dir"] = cache_dir

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
        ow, "build_inference_config_from_params", lambda p: object(), raising=False
    )

    emitted = []
    worker = ow.DetectionCacheBuildWorker(
        video_path="v.mp4",
        cache_dir=str(tmp_path),
        params={},
        start_frame=0,
        end_frame=1,
    )
    worker.finished_signal = types.SimpleNamespace(emit=lambda *a: emitted.append(a))
    worker.progress_signal = types.SimpleNamespace(emit=lambda *a: None)
    worker.run()
    assert calls.get("ran") is True
    assert calls.get("closed") is True
    assert emitted and emitted[-1][0] is True
    assert emitted[-1][1] == str(tmp_path)
