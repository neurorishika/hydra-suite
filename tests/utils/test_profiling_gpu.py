from hydra_suite.utils import profiling
from hydra_suite.utils.profiling import SpanRecorder, deep_gpu_enabled


def test_deep_gpu_disabled_by_default(monkeypatch):
    monkeypatch.delenv("HYDRA_PROFILE_GPU", raising=False)
    assert deep_gpu_enabled() is False


def test_deep_gpu_reads_the_env_var(monkeypatch):
    monkeypatch.setenv("HYDRA_PROFILE_GPU", "1")
    assert deep_gpu_enabled() is True


def test_default_recorder_never_syncs(monkeypatch):
    calls = []
    monkeypatch.setattr(profiling, "_synchronize", lambda: calls.append(1))
    rec = SpanRecorder(gpu_sync=False)
    with rec.armed():
        with rec.span("forward", gpu=True):
            pass
    assert calls == []


def test_gpu_sync_recorder_syncs_only_gpu_spans(monkeypatch):
    calls = []
    monkeypatch.setattr(profiling, "_synchronize", lambda: calls.append(1))
    rec = SpanRecorder(gpu_sync=True)
    with rec.armed():
        with rec.span("host_only"):
            pass
        with rec.span("forward", gpu=True):
            pass
    assert len(calls) == 1


def test_pipeline_depth_is_forced_to_one_in_deep_mode(monkeypatch):
    from hydra_suite.core.inference.pipeline import _effective_depth

    monkeypatch.delenv("HYDRA_PROFILE_GPU", raising=False)
    assert _effective_depth(2) == 2
    monkeypatch.setenv("HYDRA_PROFILE_GPU", "1")
    assert _effective_depth(2) == 1
    assert _effective_depth(4) == 1
