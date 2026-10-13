"""Process-level CPU cap + per-thread numba binder (runtime/cpu_budget.py).

Every assertion that mutates process-global thread state runs in a fresh
interpreter so nothing leaks into the rest of the suite.
"""

import json
import os
import subprocess
import sys
import textwrap


def _run(code: str, env_extra=None) -> dict:
    env = dict(os.environ)
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OPENCV_FFMPEG_THREADS",
        "HYDRA_CPU_THREADS",
        "HYDRA_CPU_BUDGET",
    ):
        env.pop(var, None)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env.update(env_extra or {})
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_capped_helper():
    from hydra_suite.runtime.cpu_budget import capped

    assert capped(8, None) == 8
    assert capped(8, 2) == 2
    assert capped(1, 4) == 1
    assert capped(0, 4) == 1


def test_never_set_none_is_a_noop():
    r = _run("""
        import json, os, torch, numba
        from threadpoolctl import threadpool_info
        from hydra_suite.runtime import cpu_budget as cb
        def snap():
            return {
                "torch": torch.get_num_threads(),
                "env": {v: os.environ.get(v) for v in cb.THREAD_ENV_VARS},
                "pools": sorted((p["filepath"], p["num_threads"]) for p in threadpool_info()),
                "numba": numba.get_num_threads(),
            }
        before = snap()
        cb.set_process_cpu_cap(None); cb.bind_thread_cpu_cap()
        after = snap()
        print(json.dumps({"same": before == after, "cap": cb.current_cpu_cap()}))
    """)
    assert r == {"same": True, "cap": None}


def test_cap_sets_env_threadpools_torch_and_restores():
    r = _run(
        """
        import json, os, torch
        from threadpoolctl import threadpool_info
        from hydra_suite.runtime import cpu_budget as cb
        t0 = torch.get_num_threads()
        cb.set_process_cpu_cap(2)
        capped = {
          "env": {v: os.environ.get(v) for v in cb.THREAD_ENV_VARS},
          "torch": torch.get_num_threads(),
          "pools": max(p["num_threads"] for p in threadpool_info()),
          "cap": cb.current_cpu_cap(),
        }
        cb.set_process_cpu_cap(None)
        restored = {"torch": torch.get_num_threads() == t0,
                    "env": all(os.environ.get(v) == ("64" if v == "OMP_NUM_THREADS" else None)
                               for v in cb.THREAD_ENV_VARS),
                    "cap": cb.current_cpu_cap()}
        print(json.dumps({"capped": capped, "restored": restored}))
    """,
        env_extra={"OMP_NUM_THREADS": "64"},
    )
    assert set(r["capped"]["env"].values()) == {"2"}
    assert r["capped"]["torch"] <= 2
    assert r["capped"]["pools"] <= 2
    assert r["capped"]["cap"] == 2
    assert r["restored"] == {"torch": True, "env": True, "cap": None}


def test_cap_never_raises_existing_lower_env():
    r = _run(
        """
        import json, os
        from hydra_suite.runtime import cpu_budget as cb
        cb.set_process_cpu_cap(8)
        print(json.dumps({"omp": os.environ["OMP_NUM_THREADS"]}))
    """,
        env_extra={"OMP_NUM_THREADS": "1"},
    )
    assert r == {"omp": "1"}


def test_recapping_uses_originals_not_previous_cap():
    r = _run(
        """
        import json, os
        from hydra_suite.runtime import cpu_budget as cb
        cb.set_process_cpu_cap(2); cb.set_process_cpu_cap(6)
        print(json.dumps({"omp": os.environ["OMP_NUM_THREADS"], "cap": cb.current_cpu_cap()}))
    """,
        env_extra={"OMP_NUM_THREADS": "12"},
    )
    assert r == {"omp": "6", "cap": 6}


def test_bind_is_thread_local_numba():
    r = _run("""
        import json, threading, numba
        from hydra_suite.runtime import cpu_budget as cb
        base = numba.get_num_threads()
        cb.set_process_cpu_cap(2)
        seen = {}
        def work():
            cb.bind_thread_cpu_cap(); seen["bound"] = numba.get_num_threads()
        t = threading.Thread(target=work); t.start(); t.join()
        print(json.dumps({"bound": seen["bound"], "main_untouched": numba.get_num_threads() == base}))
    """)
    assert r["bound"] <= 2 and r["main_untouched"] is True


def test_pipeline_style_thread_target_sees_cap():
    """C1: a pipeline producer thread (bind_target-wrapped closure) that binds
    first thing runs numba under the cap, even though the cap was set on the
    main thread before the producer existed."""
    r = _run("""
        import json, threading, numba
        from hydra_suite.runtime import cpu_budget as cb
        from hydra_suite.utils.profiling import bind_target
        cb.set_process_cpu_cap(2)
        seen = {}
        def producer():
            cb.bind_thread_cpu_cap()
            seen["n"] = numba.get_num_threads()
        t = threading.Thread(target=bind_target(producer), name="pipeline-obb-producer")
        t.start(); t.join()
        print(json.dumps({"n": seen["n"]}))
    """)
    assert 1 <= r["n"] <= 2


def test_bind_after_lift_restores_thread_numba():
    """A long-lived thread bound under a cap returns to its own original numba
    count once the cap is lifted and it binds again."""
    r = _run("""
        import json, numba
        from hydra_suite.runtime import cpu_budget as cb
        base = numba.get_num_threads()
        cb.set_process_cpu_cap(1); cb.bind_thread_cpu_cap()
        capped = numba.get_num_threads()
        cb.set_process_cpu_cap(None); cb.bind_thread_cpu_cap()
        print(json.dumps({"capped": capped, "restored": numba.get_num_threads() == base}))
    """)
    assert r == {"capped": 1, "restored": True}


def test_run_bgsub_batch_binds_the_calling_thread(monkeypatch):
    from hydra_suite.core.inference.stages import bgsub

    calls = []
    monkeypatch.setattr(bgsub, "bind_thread_cpu_cap", lambda: calls.append(1))
    out = bgsub.run_bgsub_batch([], [], None, bgsub.BgSubConfig(params={}), None)
    assert out == [] and calls == [1]


def test_numba_env_never_desyncs_from_numba_config():
    """CRITICAL regression: numba re-reads NUMBA_* env on every compile and
    raises once its pool is launched if NUMBA_NUM_THREADS changed. Cap, bind,
    compile, prange on a bound thread, lift, compile, re-cap, compile."""
    r = _run("""
        import json, os, threading
        import numba
        import numpy as np
        from numba import njit, prange
        from hydra_suite.runtime import cpu_budget as cb
        cb.set_process_cpu_cap(2); cb.bind_thread_cpu_cap()
        @njit(cache=False)
        def f1(x): return x + 1
        assert f1(1) == 2
        @njit(parallel=True, cache=False)
        def psum(a):
            s = 0.0
            for i in prange(a.shape[0]):
                s += a[i]
            return s
        seen = {}
        def work():
            cb.bind_thread_cpu_cap()
            seen["sum"] = psum(np.ones(1000)); seen["n"] = numba.get_num_threads()
        t = threading.Thread(target=work); t.start(); t.join()
        cb.set_process_cpu_cap(None)
        @njit(cache=False)
        def f2(x): return x * 2
        assert f2(2) == 4
        cb.set_process_cpu_cap(6)
        @njit(cache=False)
        def f3(x): return x - 1
        assert f3(2) == 1
        print(json.dumps({"sum": seen["sum"], "n": seen["n"],
                          "env": os.environ.get("NUMBA_NUM_THREADS"),
                          "cfg": numba.config.NUMBA_NUM_THREADS}))
    """)
    assert r["sum"] == 1000.0 and r["n"] <= 2
    assert r["env"] is None  # numba was loaded before the cap: env untouched


def test_numba_env_frozen_once_numba_imported_after_cap():
    r = _run("""
        import json, os
        from hydra_suite.runtime import cpu_budget as cb
        cb.set_process_cpu_cap(2)
        import numba
        from numba import njit
        cb.bind_thread_cpu_cap()
        @njit(cache=False)
        def f1(x): return x + 1
        f1(1)
        cb.set_process_cpu_cap(6)
        @njit(cache=False)
        def f2(x): return x + 2
        f2(1)
        cb.set_process_cpu_cap(None)
        @njit(cache=False)
        def f3(x): return x + 3
        f3(1)
        print(json.dumps({"env": os.environ.get("NUMBA_NUM_THREADS"),
                          "cfg": numba.config.NUMBA_NUM_THREADS}))
    """)
    assert r == {"env": "2", "cfg": 2}


def test_cap_never_raises_a_prelowered_threadpool():
    r = _run("""
        import json, torch
        from threadpoolctl import ThreadpoolController, threadpool_info
        from hydra_suite.runtime import cpu_budget as cb
        torch.get_num_threads()  # torch's lazy pool init resets OpenMP
        keep = ThreadpoolController().limit(limits=1)
        cb.set_process_cpu_cap(8)
        capped = max(p["num_threads"] for p in threadpool_info())
        cb.set_process_cpu_cap(None)
        lifted = max(p["num_threads"] for p in threadpool_info())
        print(json.dumps({"capped": capped, "lifted": lifted}))
    """)
    assert r == {"capped": 1, "lifted": 1}


def test_cap_never_raises_a_prelowered_threadpool_without_torch():
    """Without torch nothing re-lowers OpenMP after the threadpoolctl call, so
    this pins that the limiter itself never raises a smaller pool."""
    r = _run("""
        import json, sys
        import sklearn.utils._openmp_helpers  # loads an OpenMP runtime
        from threadpoolctl import ThreadpoolController, threadpool_info
        from hydra_suite.runtime import cpu_budget as cb
        keep = ThreadpoolController().limit(limits=1)
        cb.set_process_cpu_cap(8)
        capped = max(p["num_threads"] for p in threadpool_info())
        cb.set_process_cpu_cap(None)
        lifted = max(p["num_threads"] for p in threadpool_info())
        print(json.dumps({"torch": "torch" in sys.modules,
                          "capped": capped, "lifted": lifted}))
    """)
    assert r == {"torch": False, "capped": 1, "lifted": 1}


def test_torch_original_is_sticky_across_rearm():
    r = _run("""
        import json, torch
        from hydra_suite.runtime import cpu_budget as cb
        t0 = torch.get_num_threads()
        cb.set_process_cpu_cap(2); cb.set_process_cpu_cap(None)
        cb.set_process_cpu_cap(3); mid = torch.get_num_threads()
        cb.set_process_cpu_cap(None)
        print(json.dumps({"mid": mid, "back": torch.get_num_threads() == t0}))
    """)
    assert r["mid"] <= 3 and r["back"] is True


def _record_bind_threads(monkeypatch, module):
    import threading

    names = []
    monkeypatch.setattr(
        module,
        "bind_thread_cpu_cap",
        lambda: names.append(threading.current_thread().name),
    )
    return names


def _fake_pipeline(depth, cuda_mode):
    from types import SimpleNamespace

    from hydra_suite.core.inference.pipeline import Pipeline

    pipe = Pipeline.for_test(window_size=2, depth=depth, stage=lambda w: [])
    pipe.runtime = SimpleNamespace(cuda_mode=cuda_mode)
    pipe._run_detection_for_window = lambda window: []
    pipe._process_obb_results = lambda window, raw_list: list(window.frames)
    return pipe


def test_pipeline_producer_thread_binds(monkeypatch):
    from hydra_suite.core.inference import pipeline as pipeline_module

    names = _record_bind_threads(monkeypatch, pipeline_module)
    pipe = _fake_pipeline(depth=2, cuda_mode=False)
    result = pipe.run([(i, None) for i in range(5)], range(5), range_total=5)
    assert result.frames_processed == 5
    assert names == ["pipeline-obb-producer"]


def test_pipeline_decoder_and_detector_threads_bind(monkeypatch):
    from hydra_suite.core.inference import pipeline as pipeline_module

    names = _record_bind_threads(monkeypatch, pipeline_module)
    pipe = _fake_pipeline(depth=2, cuda_mode=True)
    result = pipe.run([(i, None) for i in range(5)], range(5), range_total=5)
    assert result.frames_processed == 5
    assert sorted(names) == ["pipeline-decode-producer", "pipeline-obb-producer"]


def test_run_tracking_binds_first(monkeypatch):
    """The bind is the FIRST statement: raising from it aborts before the
    method touches ``self`` at all."""
    from types import SimpleNamespace

    import pytest

    from hydra_suite.core.tracking import worker as worker_module

    class _Bound(Exception):
        pass

    def _raise():
        raise _Bound

    monkeypatch.setattr(worker_module, "bind_thread_cpu_cap", _raise)
    with pytest.raises(_Bound):
        worker_module.TrackingEngineCore.run_tracking(SimpleNamespace())
