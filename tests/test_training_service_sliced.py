from pathlib import Path

import cv2
import numpy as np

from hydra_suite.training.geometry_levels import GeometryLevel
from hydra_suite.training.service import TrainingOrchestrator
from hydra_suite.training.sliced_dataset import SliceBuildParams


def _merged(tmp: Path) -> Path:
    root = tmp / "merged"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(root / "images" / split / "f0.jpg"), np.zeros((512, 512, 3), np.uint8)
        )
        (root / "labels" / split / "f0.txt").write_text(
            "0 0.14 0.14 0.24 0.14 0.24 0.24 0.14 0.24\n", encoding="utf-8"
        )
    (root / "dataset.yaml").write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: object\n",
        encoding="utf-8",
    )
    return root


def test_orchestrator_builds_sliced_dataset(tmp_path):
    orch = TrainingOrchestrator(tmp_path / "ws")
    merged = _merged(tmp_path)
    params = SliceBuildParams(
        geometry_mode="custom",
        slice_width=256,
        slice_height=256,
        target_sizes=[],
        full_frame_mix=False,
        negative_tile_fraction=0.0,
    )
    result = orch.build_sliced_obb_dataset(
        str(merged), level=GeometryLevel.OBB, params=params, seed=7
    )
    assert Path(result.dataset_dir, "dataset.yaml").exists()
    assert Path(result.dataset_dir).is_relative_to((tmp_path / "ws").resolve())
