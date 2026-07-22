"""PoseKit's load_pose_backend shim must route a "vitpose" family to a
ViTPoseBackend (not misroute it into the SLEAP else-branch)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hydra_suite.core.identity.pose.backends.vitpose import ViTPoseBackend
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose
from hydra_suite.core.inference.api import load_pose_backend


def _tiny_ckpt(tmp_path: Path, k: int = 4) -> Path:
    model = build_vitpose("S", "classic", num_keypoints=k)
    p = tmp_path / "best.pt"
    torch.save(
        {"model_state": model.state_dict(), "variant": "S", "num_keypoints": k}, p
    )
    return p


def test_load_pose_backend_builds_vitpose(tmp_path):
    ckpt = _tiny_ckpt(tmp_path, k=4)
    backend = load_pose_backend(
        backend_family="vitpose",
        model_path=str(ckpt),
        compute_runtime="cpu",
        keypoint_names=["a", "b", "c", "d"],
        skeleton_edges=[],
        min_valid_confidence=0.0,
        batch_size=8,
        vitpose_batch=2,
        out_root=str(tmp_path),
    )
    assert isinstance(backend, ViTPoseBackend)
    assert backend.output_keypoint_names == ["a", "b", "c", "d"]
    out = backend.predict_batch([np.zeros((60, 40, 3), np.uint8)])
    assert len(out) == 1
    assert out[0].num_keypoints == 4
    backend.close()
