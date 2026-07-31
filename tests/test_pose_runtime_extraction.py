import numpy as np
import pytest


def test_onnx_runner_runs_tiny_model(tmp_path):
    pytest.importorskip("onnxruntime")
    import onnx
    from onnx import TensorProto, helper

    from hydra_suite.core.identity.pose.runtime.onnx_session import OnnxSessionRunner
    from hydra_suite.runtime.resolver import ResolvedBackend

    # identity ONNX: input (1,3,4,4) -> output same
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "id", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    p = tmp_path / "id.onnx"
    onnx.save(model, str(p))

    runner = OnnxSessionRunner(p, ResolvedBackend("torch", "cpu", False))
    out = runner.run(np.zeros((1, 3, 4, 4), np.float32))
    # returns a dict of output name -> array
    arr = next(iter(out.values())) if isinstance(out, dict) else out[0]
    assert np.asarray(arr).shape == (1, 3, 4, 4)


def test_sleap_still_imports_after_extraction():
    # SLEAP must keep working: its module imports the moved runners
    import importlib

    importlib.import_module("hydra_suite.core.identity.pose.backends.sleap")
    from hydra_suite.core.identity.pose.runtime.accelerated import (
        build_accelerated_runner,
    )
    from hydra_suite.core.identity.pose.runtime.onnx_session import OnnxSessionRunner
    from hydra_suite.core.identity.pose.runtime.tensorrt_engine import (
        TensorRTEngineRunner,
        build_trt_engine_from_onnx,
    )

    assert OnnxSessionRunner and TensorRTEngineRunner
    assert build_trt_engine_from_onnx and build_accelerated_runner
