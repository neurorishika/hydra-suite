from pathlib import Path

import numpy as np
import torch

from hydra_suite.core.identity.pose.backends.vitpose import ViTPoseBackend
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose


def _ckpt(tmp_path: Path, k: int = 4) -> Path:
    model = build_vitpose("S", "classic", num_keypoints=k)
    torch.save(
        {"model_state": model.state_dict(), "variant": "S", "num_keypoints": k},
        tmp_path / "m.pt",
    )
    return tmp_path / "m.pt"


def test_native_predict_batch_shapes(tmp_path):
    path = _ckpt(tmp_path, k=4)
    be = ViTPoseBackend(str(path), device="cpu", keypoint_names=["a", "b", "c", "d"])
    be.warmup()
    crops = [np.zeros((70, 50, 3), np.uint8), np.zeros((90, 60, 3), np.uint8)]
    results = be.predict_batch(crops)
    assert len(results) == 2
    assert results[0].num_keypoints == 4
    assert results[0].keypoints.shape == (4, 3)  # x, y, conf
    assert be.preferred_input_size == 256
    assert be.output_keypoint_names == ["a", "b", "c", "d"]
    be.close()


def test_native_empty_batch(tmp_path):
    be = ViTPoseBackend(str(_ckpt(tmp_path)), device="cpu")
    assert be.predict_batch([]) == []
