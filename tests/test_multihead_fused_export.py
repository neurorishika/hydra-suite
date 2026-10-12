"""Shared-trunk classifiers export their parallel heads as one fused MLP.

TensorRT 10.13-10.16 cannot build ViT-B/16 + two parallel GELU MLP heads +
concat in fp16 at batch > 1 (CUDA 716 / "compile resulted in null program"),
while the mathematically identical single block-diagonal MLP builds and runs
at ~3200 crops/s. The fused module must reproduce the original logits exactly.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from hydra_suite.runtime.resolver import ResolvedBackend
from hydra_suite.training.multihead_torchvision_model import (
    MultiHeadTorchvisionClassifier,
)


def _toy_multihead(widths=(9, 8), hidden=16, feat=12):
    torch.manual_seed(0)
    backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, feat))
    model = MultiHeadTorchvisionClassifier(backbone, feat, list(widths), hidden, 0.1)
    return model.eval()


def test_fused_for_export_matches_forward_exactly():
    model = _toy_multihead(widths=(9, 8, 3))
    fused = model.fused_for_export().eval()
    x = torch.randn(5, 3, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(fused(x), model(x), rtol=0, atol=1e-6)
    assert not any(isinstance(m, nn.ModuleList) for m in fused.modules())


def test_fused_head_keeps_factors_independent():
    """A factor's logits must not depend on another factor's hidden units."""
    model = _toy_multihead(widths=(4, 2), hidden=6)
    fused = model.fused_for_export()
    second = fused.head[-1].weight
    assert torch.count_nonzero(second[:4, 6:]) == 0
    assert torch.count_nonzero(second[4:, :6]) == 0


def test_onnx_export_of_multihead_has_no_parallel_heads(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    from hydra_suite.training.torchvision_model import (
        TORCHVISION_ONNX_EXPORT_VERSION,
        export_torchvision_to_onnx,
        onnx_export_version,
    )

    model = _toy_multihead()
    path = export_torchvision_to_onnx(
        model, {"input_size": (8, 8)}, tmp_path / "m.onnx"
    )
    graph = onnx.load(str(path)).graph
    assert not any(n.op_type == "Concat" for n in graph.node)
    assert onnx_export_version(path) == TORCHVISION_ONNX_EXPORT_VERSION

    x = np.random.default_rng(0).standard_normal((4, 3, 8, 8)).astype(np.float32)
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
        None, {"input": x}
    )[0]
    with torch.no_grad():
        want = model(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(got, want, atol=1e-5)


@pytest.mark.skipif(
    os.environ.get("HYDRA_OFFLINE_TESTS") == "1",
    reason="needs torchvision pretrained weights",
)
def test_stale_onnx_peer_is_re_exported(tmp_path):
    """A peer written by an older exporter (two parallel heads, no version
    stamp) sits next to existing models; it must be rebuilt, not reused."""
    from hydra_suite.core.individual.classification.backend import ClassifierBackend
    from hydra_suite.training.multihead_torchvision_model import (
        build_multihead_torchvision_classifier,
    )
    from hydra_suite.training.torchvision_model import (
        TORCHVISION_ONNX_EXPORT_VERSION,
        onnx_export_version,
        save_torchvision_checkpoint,
    )

    cnpf = [["a", "b"], ["x", "y", "z"]]
    model = build_multihead_torchvision_classifier(
        backbone="resnet18",
        class_names_per_factor=cnpf,
        trainable_layers=-1,
        head_hidden_dim=32,
        head_dropout=0.0,
        input_size=64,
    )
    ckpt_path = tmp_path / "shared_trunk.pth"
    save_torchvision_checkpoint(
        model=model,
        backbone="resnet18",
        class_names=[],
        factor_names=["color1", "color2"],
        class_names_per_factor=cnpf,
        input_size=(64, 64),
        best_val_acc=0.5,
        history={"train_loss": [], "val_acc": []},
        trainable_layers=-1,
        backbone_lr_scale=0.1,
        monochrome=False,
        extra_meta={
            "head_kind": "multihead_shared_trunk",
            "head_hidden_dim": 32,
            "head_dropout": 0.0,
        },
        path=ckpt_path,
    )
    stale = ckpt_path.with_suffix(".onnx")
    torch.onnx.export(
        model.eval(),
        torch.zeros(1, 3, 64, 64),
        str(stale),
        dynamo=False,
        input_names=["input"],
        output_names=["logits"],
    )
    assert onnx_export_version(stale) is None

    backend = ClassifierBackend(
        str(ckpt_path), resolved=ResolvedBackend("torch", "cpu", False)
    )
    peer = backend._derive_onnx_peer()
    assert peer == stale
    assert onnx_export_version(peer) == TORCHVISION_ONNX_EXPORT_VERSION
