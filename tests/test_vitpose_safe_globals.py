"""Production loaders must accept mmpose-shaped checkpoints whose `meta` dict
carries numpy scalars, without ever falling back to `weights_only=False`.

This does not need the real 1GB external checkpoint: it fabricates a small
checkpoint shaped like the real ones -- a `meta` dict with a numpy scalar
alongside a real `state_dict` -- and asserts the production loader reads it
under `weights_only=True`. Before `ensure_numpy_safe_globals()` was wired into
`load_finetuned_checkpoint`, this reproduced the exact
`UnpicklingError: Unsupported global: GLOBAL numpy.core.multiarray.scalar`
seen with the real collaborator checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hydra_suite.core.identity.pose.vitpose.adapter import load_finetuned_checkpoint
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose


def _save_mmpose_shaped_ckpt(tmp_path: Path) -> Path:
    model = build_vitpose("B", "classic", num_keypoints=6)
    ckpt = {
        # mmpose convention: a meta dict carrying numpy scalars, which is what
        # weights_only=True rejects without an explicit allowlist.
        "meta": {"epoch": np.float64(1.5), "seed": np.int64(42)},
        "state_dict": model.state_dict(),
    }
    p = tmp_path / "mmpose_shaped.pth"
    torch.save(ckpt, p)
    return p


def test_production_loader_reads_mmpose_shaped_checkpoint_with_numpy_scalars(tmp_path):
    p = _save_mmpose_shaped_ckpt(tmp_path)
    model, meta = load_finetuned_checkpoint(p)
    assert meta.variant == "B"
    assert meta.head == "classic"
    assert meta.num_keypoints == 6
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 256, 192))
    assert out.shape == (1, 6, 64, 48)
