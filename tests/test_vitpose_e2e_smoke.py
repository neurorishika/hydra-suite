# tests/test_vitpose_e2e_smoke.py
"""Acceptance loop: (mini) train -> best.pt -> backend -> keypoints.

Uses the training payload's own model builder to fabricate a 'trained'
checkpoint (a real training run is covered by Spec-4 tests); this asserts the
integration seam: a training-format checkpoint tracks end to end through the
production factory.
"""

import numpy as np
import torch

from hydra_suite.core.identity.pose.api import create_pose_backend_from_config
from hydra_suite.core.identity.pose.types import PoseRuntimeConfig
from hydra_suite.core.identity.pose.vitpose.training.model_setup import (
    build_finetune_model,
)


def test_training_checkpoint_tracks_end_to_end(tmp_path):
    # 1. produce a training-format best.pt (classic head, as the trainer does)
    model = build_finetune_model(variant="S", num_keypoints=5, drop_path=0.1)
    ckpt = {
        "model_state": model.state_dict(),
        "optim_state": {},
        "variant": "S",
        "num_keypoints": 5,
        "epoch": 1,
        "pck": 0.42,
        "sched_state": {},
    }
    best = tmp_path / "best.pt"
    torch.save(ckpt, best)

    # 2. build the production backend through the factory
    cfg = PoseRuntimeConfig(
        backend_family="vitpose",
        runtime_flavor="native",
        device="cpu",
        model_path=str(best),
        keypoint_names=[f"k{i}" for i in range(5)],
    )
    backend = create_pose_backend_from_config(cfg)

    # 3. track two synthetic crops
    crops = [np.random.randint(0, 255, (80, 60, 3), np.uint8) for _ in range(2)]
    results = backend.predict_batch(crops)
    assert len(results) == 2
    assert results[0].keypoints.shape == (5, 3)
    assert np.all(np.isfinite(results[0].keypoints))
    backend.close()
