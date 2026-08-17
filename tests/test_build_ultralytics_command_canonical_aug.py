"""build_ultralytics_command threads the aug profile into the classify prefit."""

import cv2
import numpy as np

from hydra_suite.training import runner as R
from hydra_suite.training.contracts import AugmentationProfile, TrainingHyperParams


def _seed_classify_dataset(root):
    d = root / "train" / "clsA"
    d.mkdir(parents=True)
    cv2.imwrite(str(d / "a.png"), np.full((30, 50, 3), 120, np.uint8))
    return root


def test_classify_prefit_receives_profile_and_seed(tmp_path, monkeypatch):
    ds = _seed_classify_dataset(tmp_path / "ds")
    captured = {}

    def _fake_prefit(dataset_dir, imgsz, dest_dir, *, profile=None, seed=42):
        captured["profile"] = profile
        captured["seed"] = seed
        (dest_dir).mkdir(parents=True, exist_ok=True)
        return dest_dir

    monkeypatch.setattr(R, "_prefit_yolo_classify_dataset", _fake_prefit)

    prof = AugmentationProfile(canonical_aug=True, canonical_aug_copies=2)
    spec = R.TrainingRunSpec(
        role=R.TrainingRole.CLASSIFY_FLAT_YOLO,  # a YOLO classify role -> task "classify"
        source_datasets=[],
        derived_dataset_dir=str(ds),
        base_model="yolov8n-cls.pt",
        hyperparams=TrainingHyperParams(imgsz=64, epochs=1, batch=1),
        seed=99,
        augmentation_profile=prof,
    )
    R.build_ultralytics_command(spec, tmp_path / "run")
    assert captured["seed"] == 99
    assert captured["profile"] is prof
    assert captured["profile"].canonical_aug is True
