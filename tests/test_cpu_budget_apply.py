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
        import json, os, torch
        from hydra_suite.runtime import cpu_budget as cb
        before = (torch.get_num_threads(), os.environ.get("OMP_NUM_THREADS"))
        cb.set_process_cpu_cap(None); cb.bind_thread_cpu_cap()
        after = (torch.get_num_threads(), os.environ.get("OMP_NUM_THREADS"))
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
