import os
from pathlib import Path

import pytest
import torch

from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose
from hydra_suite.core.identity.pose.vitpose.weights import load_checkpoint

ASSET_DIR = Path(os.path.expanduser("~/.cache/vitpose-assets"))

requires_weights = pytest.mark.skipif(
    not (ASSET_DIR / "vitpose-b.pth").exists(),
    reason="run tools/vitpose/fetch_assets.py first",
)


def test_forward_shape_without_weights():
    m = build_vitpose("B", "classic").eval()
    with torch.no_grad():
        out = m(torch.zeros(1, 3, 256, 192))
    assert out.shape == (1, 17, 64, 48)


@requires_weights
def test_gate_a_strict_load_classic():
    """GATE A(1). strict=True is the architecture test: a wrong patch padding
    or a dropped pos_embed cls slot fails here with no ambiguity."""
    m = build_vitpose("B", "classic")
    load_checkpoint(m, ASSET_DIR / "vitpose-b.pth", strict=True)


@requires_weights
def test_gate_a_strict_load_simple():
    """GATE A(2)."""
    m = build_vitpose("B", "simple")
    load_checkpoint(m, ASSET_DIR / "vitpose-b-simple.pth", strict=True)


@requires_weights
def test_checkpoint_load_is_weights_only():
    """Checkpoints come from a third-party re-host. weights_only=False would
    permit arbitrary code execution via unpickling."""
    import inspect

    from hydra_suite.core.identity.pose.vitpose import weights

    src = inspect.getsource(weights)
    assert "weights_only=True" in src
    assert "weights_only=False" not in src


@requires_weights
def test_loaded_model_produces_finite_heatmaps():
    m = build_vitpose("B", "classic").eval()
    load_checkpoint(m, ASSET_DIR / "vitpose-b.pth", strict=True)
    with torch.no_grad():
        out = m(torch.zeros(1, 3, 256, 192))
    assert out.shape == (1, 17, 64, 48)
    assert torch.isfinite(out).all()
