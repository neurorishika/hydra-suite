from pathlib import Path

import pytest
import torch

from hydra_suite.core.identity.pose.backends.vitpose import (
    _vitpose_artifact_signature,
    auto_export_vitpose_model,
)
from hydra_suite.core.identity.pose.types import PoseRuntimeConfig
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose


def _ckpt(tmp_path):
    model = build_vitpose("S", "classic", num_keypoints=3)
    p = tmp_path / "best.pt"
    torch.save(
        {"model_state": model.state_dict(), "variant": "S", "num_keypoints": 3}, p
    )
    return p


def test_signature_includes_recipe_tag(tmp_path):
    p = _ckpt(tmp_path)
    sig = _vitpose_artifact_signature(str(p), "coreml")
    assert "vitpose-v1" in sig
    assert "coreml" in sig


def test_coreml_export_cached(tmp_path):
    pytest.importorskip("coremltools")
    p = _ckpt(tmp_path)
    cfg = PoseRuntimeConfig(
        backend_family="vitpose",
        model_path=str(p),
        runtime_flavor="coreml",
        device="mps",
    )
    art = auto_export_vitpose_model(cfg, "coreml")
    assert Path(art).exists()
    assert Path(art).suffix == ".mlpackage" or Path(art).name.endswith(".mlpackage")
    # second call reuses (mtime unchanged)
    mtime = Path(art).stat().st_mtime_ns
    art2 = auto_export_vitpose_model(cfg, "coreml")
    assert art2 == art
    assert Path(art).stat().st_mtime_ns == mtime  # not re-exported
