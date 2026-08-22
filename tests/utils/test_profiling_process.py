import json

import pytest

from hydra_suite.utils import profiling_process
from hydra_suite.utils.profiling import PRIORITY_PROCESS, current, span
from hydra_suite.utils.profiling_process import maybe_arm_process_recorder


@pytest.fixture(autouse=True)
def _reset():
    profiling_process.reset_for_test()
    yield
    profiling_process.reset_for_test()


def test_not_armed_without_the_env_var(monkeypatch):
    monkeypatch.delenv("HYDRA_PROFILE", raising=False)
    monkeypatch.delenv("HYDRA_RT_PROFILE", raising=False)
    assert maybe_arm_process_recorder() is None
    assert current() is None


def test_armed_by_hydra_profile(monkeypatch):
    monkeypatch.setenv("HYDRA_PROFILE", "1")
    rec = maybe_arm_process_recorder()
    assert rec is not None
    assert rec.priority == PRIORITY_PROCESS
    with span("detectkit_work"):
        pass
    assert rec.snapshot()["children"][0]["name"] == "detectkit_work"


def test_legacy_alias_still_works(monkeypatch):
    monkeypatch.delenv("HYDRA_PROFILE", raising=False)
    monkeypatch.setenv("HYDRA_RT_PROFILE", "1")
    assert maybe_arm_process_recorder() is not None


def test_arming_is_idempotent(monkeypatch):
    monkeypatch.setenv("HYDRA_PROFILE", "1")
    assert maybe_arm_process_recorder() is maybe_arm_process_recorder()


def test_tracking_profiler_wins_while_armed(monkeypatch):
    from hydra_suite.core.tracking.profiler import TrackingProfiler

    monkeypatch.setenv("HYDRA_PROFILE", "1")
    process = maybe_arm_process_recorder()
    prof = TrackingProfiler(enabled=True)
    with prof.armed():
        with span("inside_session"):
            pass
    with span("outside_session"):
        pass
    assert prof.spans.snapshot()["children"][0]["name"] == "inside_session"
    assert [c["name"] for c in process.snapshot()["children"]] == ["outside_session"]


def test_dump_writes_json(monkeypatch, tmp_path):
    monkeypatch.setenv("HYDRA_PROFILE", "1")
    monkeypatch.setattr(profiling_process, "dump_path", lambda: tmp_path / "p.json")
    maybe_arm_process_recorder()
    with span("work"):
        pass
    profiling_process.dump()
    loaded = json.loads((tmp_path / "p.json").read_text())
    assert loaded["spans"]["children"][0]["name"] == "work"


def test_rt_prof_machinery_is_gone():
    import hydra_suite.core.inference.runner as runner

    for name in ("_RT_PROF_ACC", "_rt_prof_on", "_rt_prof_add", "_rt_prof_flush"):
        assert not hasattr(runner, name), f"{name} should have been deleted"


def test_detectkit_and_posekit_paths_arm_the_recorder(monkeypatch):
    """Risk 4's mitigation must not ship untested.

    Both kits drive ``core/inference`` with no ``TrackingProfiler``, so
    ``InferenceRunner.__init__`` is the arming point they share. Assert on the
    arming seam rather than constructing a kit — the kits need Qt and models,
    which no unit test should require.
    """
    import hydra_suite.core.inference.runner as runner

    monkeypatch.setenv("HYDRA_PROFILE", "1")
    src = __import__("inspect").getsource(runner.InferenceRunner.__init__)
    assert "maybe_arm_process_recorder" in src

    rec = maybe_arm_process_recorder()
    with span("kit_driven_work"):
        pass
    assert rec.snapshot()["children"][0]["name"] == "kit_driven_work"
