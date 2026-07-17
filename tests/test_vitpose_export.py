import os
from pathlib import Path

import numpy as np
import pytest
import torch

from hydra_suite.core.identity.pose.vitpose.export import ExportError, export_onnx
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose
from hydra_suite.core.identity.pose.vitpose.weights import load_checkpoint

ASSET_DIR = Path(os.path.expanduser("~/.cache/vitpose-assets"))
requires_weights = pytest.mark.skipif(
    not (ASSET_DIR / "vitpose-b.pth").exists(),
    reason="run tools/vitpose/fetch_assets.py first",
)


def test_export_refuses_a_training_mode_model(tmp_path):
    """model.eval() is mandatory: the classic head's BatchNorm2d layers are the only
    stateful modules, and exporting in train mode emits training-mode
    BatchNormalization that silently produces garbage. Refuse loudly instead."""
    m = build_vitpose("B", "classic").train()
    with pytest.raises(ExportError, match="eval"):
        export_onnx(m, tmp_path / "x.onnx")


def test_export_refuses_a_sub_minimum_opset(tmp_path):
    """The recipe documents 'opset 14+'; a caller passing an older opset (mmpose's
    exporter historically asserted opset_version == 11) must get an ExportError naming
    the minimum, not an opaque failure deep inside torch.onnx.export or onnxruntime."""
    m = build_vitpose("B", "classic").eval()
    with pytest.raises(ExportError, match="opset"):
        export_onnx(m, tmp_path / "x.onnx", opset=11)


def test_legacy_exporter_rejection_raises_export_error(tmp_path, monkeypatch):
    """When torch.onnx.export can no longer honour dynamo=False (kwarg removed, or the
    legacy TorchScript exporter module it selects is gone), export_onnx must translate
    the raw TypeError/ImportError into an ExportError naming the cause and the two real
    remedies -- not let the upstream exception leak through unexplained.

    We cannot force a real future torch to drop the legacy exporter here, so this
    exercises the wrapper directly: monkeypatch torch.onnx.export to raise the failure
    modes we catch and assert the translation happens with a useful message.
    """
    import hydra_suite.core.identity.pose.vitpose.export as export_mod

    m = build_vitpose("B", "classic").eval()

    def _raise_type_error(*args, **kwargs):
        raise TypeError("export() got an unexpected keyword argument 'dynamo'")

    monkeypatch.setattr(export_mod.torch.onnx, "export", _raise_type_error)
    with pytest.raises(ExportError, match="legacy exporter"):
        export_onnx(m, tmp_path / "x.onnx")

    def _raise_import_error(*args, **kwargs):
        raise ImportError("No module named 'torch.onnx._internal.torchscript_exporter'")

    monkeypatch.setattr(export_mod.torch.onnx, "export", _raise_import_error)
    with pytest.raises(ExportError, match="legacy exporter"):
        export_onnx(m, tmp_path / "y.onnx")


@requires_weights
def test_gate_d_onnx_matches_torch(tmp_path):
    """GATE D(onnx). ONNX on the CPU EP is the same math at the same precision, so it
    should be near-exact. Bound is max-abs per element, not a mean -- an averaged bound
    hides a single bad output channel, which is the failure that matters."""
    import onnxruntime as ort

    m = build_vitpose("B", "classic").eval()
    load_checkpoint(m, ASSET_DIR / "vitpose-b.pth", strict=True)
    onnx_path = export_onnx(m, tmp_path / "vitpose-b.onnx")

    x = torch.randn(2, 3, 256, 192)
    with torch.no_grad():
        ref = m(x).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]

    assert got.shape == ref.shape
    assert (
        np.abs(ref - got).max() < 1e-4
    ), f"max|onnx-torch| = {np.abs(ref - got).max():.3e}"


@requires_weights
def test_onnx_honours_dynamic_batch(tmp_path):
    """forward() reshapes with ints from .shape, which trace to literals -- without
    dynamic_axes the graph is pinned to the export batch size. Export at batch 2, run
    at batch 5."""
    import onnxruntime as ort

    m = build_vitpose("B", "classic").eval()
    load_checkpoint(m, ASSET_DIR / "vitpose-b.pth", strict=True)
    onnx_path = export_onnx(m, tmp_path / "dyn.onnx", dynamic_batch=True)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(
        None, {sess.get_inputs()[0].name: np.zeros((5, 3, 256, 192), np.float32)}
    )[0]
    assert out.shape == (5, 17, 64, 48)


@requires_weights
def test_moe_export_bakes_one_expert(tmp_path):
    """Upstream's masked-sum runs all 6 experts and zeroes 5 (a DDP workaround). Exporting
    that would put 6x the expert-branch compute in the graph for no benefit. With a
    concrete dataset_index the graph must carry exactly one expert's Gemm per block.

    Asserts on the graph, not on wall-clock: timing is noisy, node counts are not.
    """
    import onnx

    from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose_moe

    m = build_vitpose_moe("B").eval()
    load_checkpoint(m, ASSET_DIR / "vitpose+_base.pth", strict=True)
    onnx_path = export_onnx(m, tmp_path / "moe.onnx", dataset_index=0)
    g = onnx.load(str(onnx_path)).graph
    # 12 blocks; a 6-expert masked sum would add >= 6 Gemm/MatMul per block over the
    # single-expert graph. Bound generously -- the point is 1x not 6x.
    gemms = sum(1 for n in g.node if n.op_type in ("Gemm", "MatMul"))
    assert gemms < 12 * 10, f"{gemms} Gemm/MatMul nodes — masked-sum likely exported"


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)


@requires_cuda
@requires_weights
def test_tensorrt_defaults_to_fp32(tmp_path):
    """FP32 is a deliberate decision, not an oversight: sleap.py:420-421 keeps FP32 'to
    preserve keypoint precision' and compute_runtime.py:141-142 states the same rule.
    A future edit flipping this default must break a test, not slip through."""
    import inspect

    from hydra_suite.core.identity.pose.vitpose import export

    sig = inspect.signature(export.build_tensorrt_engine)
    assert sig.parameters["fp16"].default is False


@requires_cuda
@requires_weights
def test_gate_d_tensorrt_matches_torch(tmp_path):
    """GATE D(tensorrt). TRT rearranges kernels, so it gets more slack than ONNX -- but
    FP32 keeps it close. Bound is max-abs per element."""
    from hydra_suite.core.identity.pose.vitpose.export import (
        build_tensorrt_engine,
        export_onnx,
    )
    from tools.vitpose.trt_runner import run_engine  # test-only helper, see Step 3

    m = build_vitpose("B", "classic").eval()
    load_checkpoint(m, ASSET_DIR / "vitpose-b.pth", strict=True)
    onnx_path = export_onnx(m, tmp_path / "b.onnx")
    engine = build_tensorrt_engine(onnx_path, tmp_path / "b.engine")

    x = torch.randn(2, 3, 256, 192)
    with torch.no_grad():
        ref = m(x).numpy()
    got = run_engine(engine, x.numpy())
    assert got.shape == ref.shape
    assert (
        np.abs(ref - got).max() < 1e-3
    ), f"max|trt-torch| = {np.abs(ref - got).max():.3e}"
