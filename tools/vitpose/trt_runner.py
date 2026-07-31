"""Test-only TensorRT engine runner.

This exists solely to feed a numpy batch through a built engine and get numpy
back so ``tests/test_vitpose_export.py`` can compare against torch. It is NOT
the production runtime: the real TensorRT execution path (device-resident I/O
reuse, batching, dtype/layout auto-detection, legacy-binding-API fallback for
pre-10.x TensorRT) is a later spec's job -- it will extract sleap.py:468's
``_DirectTensorRTEngine`` into a shared runtime component. Do not extend this
into a production wrapper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def run_engine(engine_path: Path, x: np.ndarray) -> np.ndarray:
    """Run a single-input/single-output TensorRT engine on a numpy batch.

    Assumes (as export_onnx guarantees) exactly one input and one output
    tensor, float32, with only the leading batch dim dynamic.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(
            f"Failed to create TensorRT execution context: {engine_path}"
        )

    input_name = None
    output_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_name = name
    if input_name is None or output_name is None:
        raise RuntimeError(
            f"TensorRT engine is missing an input or output tensor: {engine_path}"
        )

    context.set_input_shape(input_name, tuple(int(v) for v in x.shape))

    input_tensor = torch.as_tensor(
        np.ascontiguousarray(x), device="cuda", dtype=torch.float32
    )
    out_shape = tuple(int(v) for v in context.get_tensor_shape(output_name))
    output_tensor = torch.empty(out_shape, device="cuda", dtype=torch.float32)

    context.set_tensor_address(input_name, int(input_tensor.data_ptr()))
    context.set_tensor_address(output_name, int(output_tensor.data_ptr()))
    stream = torch.cuda.current_stream().cuda_stream
    ok = context.execute_async_v3(stream_handle=stream)
    if not ok:
        raise RuntimeError(f"TensorRT inference failed: {engine_path}")
    torch.cuda.current_stream().synchronize()
    return output_tensor.detach().cpu().numpy()
