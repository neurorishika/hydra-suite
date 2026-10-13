"""End-to-end probe: a child process really runs under the CPU cap.

Every assertion runs in a real subprocess, so nothing here touches the test
process's own thread pools. Three legs:

* ``fanout``  -- the env ``build_child_env(threads_per_job=2)`` hands a fan-out
  child. Its thread env vars already shrink most pools at library load, so this
  leg proves the env hand-off end to end, plus the ORT intra-op option (which
  no env var controls).
* ``budget``  -- every thread env var and ``HYDRA_CPU_THREADS`` stripped, only
  ``HYDRA_CPU_BUDGET=2`` set (the single-run path). Here nothing but the
  in-process ``single_run_cpu_cap``/``set_process_cpu_cap``/
  ``bind_thread_cpu_cap`` + ``session_options_for`` can lower the pools, so a
  no-op cap fails this leg.
* ``uncapped`` -- the plain path: no cap, ORT defaults.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from hydra_suite.runtime.cpu_budget import THREAD_ENV_VARS, usable_cores
from hydra_suite.trackerkit.batch_fanout import build_child_env

_CAP = 2

# Mirrors a real child: same import graph, the same process-cap resolution the
# CLI does in ``cli.run_tracking_cli``, then the decode + ORT work a tracking
# child does. ORT sessions are built exactly as the pipeline builds them
# (``session_options_for(current_cpu_cap())`` -> ORT defaults when uncapped).
_CHILD = textwrap.dedent("""
    import json, sys, time

    import numba
    import hydra_suite.trackerkit.app  # same import graph as a real child
    from threadpoolctl import threadpool_info
    from hydra_suite.runtime import cpu_budget as cb

    cb.set_process_cpu_cap(cb.single_run_cpu_cap(None))
    cb.bind_thread_cpu_cap()

    import cv2
    import numpy as np
    import onnxruntime as ort
    import psutil
    import torch
    from hydra_suite.runtime.onnx_providers import session_options_for

    video, model = sys.argv[1], sys.argv[2]
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), video
    frames = 0
    for _ in range(10):
        ok, _frame = cap.read()
        frames += int(ok)

    sess = ort.InferenceSession(
        model,
        sess_options=session_options_for(cb.current_cpu_cap()),
        providers=["CPUExecutionProvider"],
    )
    sess.run(None, {"x": np.ones((256, 256), dtype=np.float32)})

    torch.mm(torch.ones(64, 64), torch.ones(64, 64))
    time.sleep(0.2)  # let lazily spawned worker threads settle
    print(json.dumps({
        "cap": cb.current_cpu_cap(),
        "pools": max((p["num_threads"] for p in threadpool_info()), default=0),
        "numba": numba.get_num_threads(),
        "torch": torch.get_num_threads(),
        "ort_intra_op": sess.get_session_options().intra_op_num_threads,
        "os_threads": psutil.Process().num_threads(),
        "frames": frames,
    }))
    cap.release()
""")


def _make_clip(path) -> None:
    av = pytest.importorskip("av")
    import numpy as np

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=25)
        stream.width, stream.height = 640, 480
        stream.pix_fmt = "yuv420p"
        rng = np.random.default_rng(0)
        for _ in range(30):
            img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _make_onnx(path) -> str:
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    weight = numpy_helper.from_array(
        np.full((256, 256), 0.01, dtype=np.float32), name="w"
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        "probe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [256, 256])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [256, 256])],
        initializer=[weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return str(path)


def _probe(env, video, model) -> dict:
    out = subprocess.run(
        [sys.executable, "-c", _CHILD, str(video), model],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _base_env() -> dict:
    env = dict(os.environ)
    # A pristine parent: no inherited budget or cap, no pre-set thread vars.
    for var in ("HYDRA_CPU_BUDGET", "HYDRA_CPU_THREADS", *THREAD_ENV_VARS):
        env.pop(var, None)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # Pin the child to the hydra_suite this test imported, never an editable
    # install of another checkout.
    import hydra_suite

    src = str(Path(hydra_suite.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (src, env.get("PYTHONPATH", "")) if p
    )
    return env


def _assert_capped(r: dict) -> None:
    assert r["frames"] > 0
    assert r["cap"] == _CAP
    assert r["pools"] <= _CAP
    assert r["numba"] <= _CAP
    assert r["ort_intra_op"] == _CAP


@pytest.mark.skipif((os.cpu_count() or 1) < 4, reason="needs >= 4 cores")
def test_capped_child_pools_respect_cap(tmp_path):
    video = tmp_path / "probe.mp4"
    _make_clip(video)
    model = _make_onnx(tmp_path / "probe.onnx")

    fanout_env = build_child_env(_base_env(), gpu=None, threads_per_job=_CAP)
    budget_env = _base_env()
    budget_env["HYDRA_CPU_BUDGET"] = str(_CAP)
    uncapped_env = build_child_env(_base_env(), gpu=None, threads_per_job=None)
    assert "HYDRA_CPU_THREADS" not in uncapped_env
    assert not any(v in budget_env for v in (*THREAD_ENV_VARS, "HYDRA_CPU_THREADS"))

    fanout = _probe(fanout_env, video, model)
    budget = _probe(budget_env, video, model)
    uncapped = _probe(uncapped_env, video, model)
    # torch and OS thread counts are logged; torch's pool size is
    # import-order dependent and OS threads are a lazy, host-dependent proxy.
    print("fanout child probe:", fanout)
    print("budget-only child probe:", budget)
    print("uncapped child probe:", uncapped)

    _assert_capped(fanout)
    _assert_capped(budget)  # only the in-process cap can deliver this leg

    assert uncapped["frames"] > 0
    assert uncapped["cap"] is None
    assert uncapped["ort_intra_op"] == 0  # ORT default: untouched plain path

    # The OS-thread margin is only meaningful on a box with many cores; on
    # small hosts the capped/uncapped gap is within run-to-run noise.
    if usable_cores() >= 12:
        for leg in (fanout, budget):
            assert leg["os_threads"] < 0.6 * uncapped["os_threads"], (
                leg,
                uncapped,
            )
