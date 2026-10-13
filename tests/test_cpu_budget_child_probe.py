"""End-to-end probe: a fan-out child built with ``threads_per_job`` really runs capped.

Every assertion runs in a real subprocess spawned with the env that
``build_child_env`` hands a fan-out child, so nothing here touches the test
process's own thread pools.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

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

    cb.set_process_cpu_cap(cb.resolve_process_cap(None))
    cb.bind_thread_cpu_cap()

    import cv2
    import numpy as np
    import psutil
    import torch

    video, model = sys.argv[1], sys.argv[2]
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), video
    frames = 0
    for _ in range(10):
        ok, _frame = cap.read()
        frames += int(ok)

    ort_ran = False
    if model:
        import onnxruntime as ort
        from hydra_suite.runtime.onnx_providers import session_options_for

        sess = ort.InferenceSession(
            model,
            sess_options=session_options_for(cb.current_cpu_cap()),
            providers=["CPUExecutionProvider"],
        )
        x = np.ones((256, 256), dtype=np.float32)
        sess.run(None, {"x": x})
        ort_ran = True

    torch.mm(torch.ones(64, 64), torch.ones(64, 64))
    time.sleep(0.2)  # let lazily spawned worker threads settle
    print(json.dumps({
        "cap": cb.current_cpu_cap(),
        "pools": max((p["num_threads"] for p in threadpool_info()), default=0),
        "numba": numba.get_num_threads(),
        "torch": torch.get_num_threads(),
        "os_threads": psutil.Process().num_threads(),
        "frames": frames,
        "ort": ort_ran,
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
    try:
        import numpy as np
        import onnx
        import onnxruntime  # noqa: F401
        from onnx import TensorProto, helper, numpy_helper
    except ImportError:
        return ""
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
    for var in ("HYDRA_CPU_BUDGET", "HYDRA_CPU_THREADS"):
        env.pop(var, None)
    from hydra_suite.runtime.cpu_budget import THREAD_ENV_VARS

    for var in THREAD_ENV_VARS:
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


@pytest.mark.skipif((os.cpu_count() or 1) < 4, reason="needs >= 4 cores")
def test_capped_child_pools_respect_cap(tmp_path):
    video = tmp_path / "probe.mp4"
    _make_clip(video)
    model = _make_onnx(tmp_path / "probe.onnx")

    capped_env = build_child_env(_base_env(), gpu=None, threads_per_job=_CAP)
    uncapped_env = build_child_env(_base_env(), gpu=None, threads_per_job=None)
    assert "HYDRA_CPU_THREADS" not in uncapped_env

    capped = _probe(capped_env, video, model)
    uncapped = _probe(uncapped_env, video, model)
    # torch is logged, not asserted (its pool size is import-order dependent).
    print("capped child probe:", capped)
    print("uncapped child probe:", uncapped)

    assert capped["frames"] > 0 and uncapped["frames"] > 0
    assert capped["cap"] == _CAP
    assert uncapped["cap"] is None
    assert capped["pools"] <= _CAP
    assert capped["numba"] <= _CAP
    assert capped["os_threads"] < 0.6 * uncapped["os_threads"], (capped, uncapped)
