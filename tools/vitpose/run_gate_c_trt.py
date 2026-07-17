"""Task 13 Step 5: run Gate C (COCO val AP) through a built TensorRT engine.

This is a standalone runner, not a test, so it can log wall-clock and engine
provenance while it runs. It builds a real FP32 TensorRT engine from the same
export recipe Gate D(tensorrt) already validates (src/.../vitpose/export.py,
untouched here), then feeds every forward pass in tools/vitpose/eval_coco.py's
Gate C harness through that engine via the harness's ``forward_fn`` injection
point -- the flip test, UDP decode, transform_preds, rescoring, and OKS-NMS
are reused verbatim from eval_coco.evaluate(), not duplicated.

Usage:
    python tools/vitpose/run_gate_c_trt.py [classic|simple] [--limit N] [--batch-size N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hydra_suite.core.identity.pose.vitpose.export import (  # noqa: E402
    build_tensorrt_engine,
    export_onnx,
)
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose  # noqa: E402
from hydra_suite.core.identity.pose.vitpose.weights import load_checkpoint  # noqa: E402
from tools.vitpose.eval_coco import evaluate  # noqa: E402

ASSET_DIR = Path(os.path.expanduser("~/.cache/vitpose-assets"))

_CONFIGS = {
    "classic": ("B", "classic", ASSET_DIR / "vitpose-b.pth", 75.8),
    "simple": ("B", "simple", ASSET_DIR / "vitpose-b-simple.pth", 75.5),
}


class TrtForward:
    """Persistent TensorRT execution context wrapping a single engine.

    ``trt_runner.run_engine`` deserializes the engine fresh on every call --
    fine for the one-shot Gate D parity test, ruinous for a ~3,900-image eval
    (thousands of forward calls). This keeps one engine + one execution
    context alive for the whole run and reuses them, and counts every call so
    the report can prove the engine (not a silent torch fallback) served every
    forward pass.
    """

    def __init__(self, engine_path: Path, device: str = "cuda") -> None:
        import tensorrt as trt

        assert device == "cuda", "TensorRT execution requires a CUDA device"
        self.device = device
        self.calls = 0
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        if self.input_name is None or self.output_name is None:
            raise RuntimeError("Engine is missing an input or output tensor")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        x = x.contiguous().to(self.device, dtype=torch.float32)
        self.context.set_input_shape(self.input_name, tuple(x.shape))
        out_shape = tuple(
            int(v) for v in self.context.get_tensor_shape(self.output_name)
        )
        out = torch.empty(out_shape, device=self.device, dtype=torch.float32)
        self.context.set_tensor_address(self.input_name, int(x.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(out.data_ptr()))
        stream = torch.cuda.current_stream().cuda_stream
        ok = self.context.execute_async_v3(stream_handle=stream)
        if not ok:
            raise RuntimeError("TensorRT inference failed")
        torch.cuda.current_stream().synchronize()
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "head", choices=["classic", "simple"], default="classic", nargs="?"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    assert (
        torch.cuda.is_available()
    ), "This runner requires CUDA (TensorRT engine target)"
    device = "cuda"

    variant, head, ckpt, target = _CONFIGS[args.head]

    work = Path("/tmp/vitpose-gate-c-trt")
    work.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Exporting ONNX + building FP32 TensorRT engine for {args.head} ...")
    t0 = time.time()
    model = build_vitpose(variant, head).eval()
    load_checkpoint(model, ckpt, strict=True)
    onnx_path = export_onnx(model, work / f"{args.head}.onnx")
    engine_path = build_tensorrt_engine(
        onnx_path, work / f"{args.head}.engine", fp16=False, max_batch=64
    )
    print(f"    engine built at {engine_path} in {time.time() - t0:.1f}s")

    forward = TrtForward(engine_path, device=device)

    print(
        f"[2/3] Running Gate C eval through the engine (batch_size={args.batch_size}) ..."
    )
    t0 = time.time()
    res = evaluate(
        variant,
        head,
        ckpt,
        device=device,
        limit=args.limit,
        batch_size=args.batch_size,
        forward_fn=forward,
    )
    elapsed = time.time() - t0

    print(f"[3/3] Done in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"engine forward() calls: {forward.calls}")
    print(f"AP = {res['AP'] * 100:.2f}  (target {target} +/- 0.2)")
    print(f"delta = {res['AP'] * 100 - target:+.2f}")


if __name__ == "__main__":
    main()
