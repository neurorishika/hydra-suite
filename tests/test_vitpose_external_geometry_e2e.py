"""The external 256x256 checkpoint must load through the production path.

Skipped unless the 1 GB checkpoint is present; it is not in the repo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

CKPT = Path(
    "/Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker"
    "/.worktrees/vitpose_external/ViTPose_base_ant9kp_256x256.pth"
)

pytestmark = pytest.mark.skipif(
    not CKPT.exists(), reason="external ViTPose checkpoint not present"
)


def test_external_checkpoint_loads_with_square_geometry():
    from hydra_suite.core.individual.pose.vitpose.adapter import (
        load_finetuned_checkpoint,
    )
    from hydra_suite.core.individual.pose.vitpose.geometry import PoseGeometry

    model, meta = load_finetuned_checkpoint(CKPT)
    assert meta.geometry == PoseGeometry((256, 256))
    assert meta.num_keypoints == 9
    assert meta.head == "classic"
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 256, 256))
    assert out.shape == (1, 9, 64, 64)


def test_production_loader_rebuilds_the_same_model_as_the_probe():
    """The probe's standalone loader is the validated reference. Given the same
    input tensor, the production-loaded model must produce the same heatmaps.

    Compare at the HEATMAP, not at final coordinates: the two paths preprocess
    differently on purpose (the probe warps the crop straight to 256x256, while
    production applies box2cs with PADDING_FACTOR=1.25) and decode differently
    on purpose (mmpose-'default' quarter-offset vs UDP). Feeding both the same
    tensor isolates what this slice actually changed -- checkpoint loading and
    model construction -- from those deliberate differences.
    """
    from hydra_suite.core.individual.pose.vitpose.adapter import (
        load_finetuned_checkpoint,
    )
    from tools.vitpose.external_ckpt.model import load_external_checkpoint, preprocess

    rng = np.random.default_rng(0)
    crop = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    batch = torch.from_numpy(preprocess(crop)[None]).float()

    probe_model, _ = load_external_checkpoint(CKPT)
    prod_model, meta = load_finetuned_checkpoint(CKPT)

    with torch.no_grad():
        probe_out = probe_model.eval()(batch)
        prod_out = prod_model.eval()(batch)

    assert prod_out.shape == probe_out.shape == (1, 9, 64, 64)
    assert torch.allclose(prod_out, probe_out, atol=1e-5)


def test_production_preprocess_uses_the_checkpoint_geometry():
    from hydra_suite.core.individual.pose.vitpose.adapter import (
        load_finetuned_checkpoint,
    )
    from hydra_suite.core.individual.pose.vitpose.infer import preprocess_crop

    _, meta = load_finetuned_checkpoint(CKPT)
    chw, _, _ = preprocess_crop(
        np.zeros((120, 120, 3), dtype=np.uint8), geom=meta.geometry
    )
    assert chw.shape == (3, 256, 256)
