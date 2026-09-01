"""The role must be reachable: no raise, no ultralytics fall-through."""

import json

from hydra_suite.training.contracts import TrainingRole
from hydra_suite.training.dataset_builders import role_min_level
from hydra_suite.training.geometry_levels import GeometryLevel
from hydra_suite.training.validation import validate_role_dataset


def _coco(tmp_path, n_images=2, n_anns=3):
    for split in ("train", "valid"):
        d = tmp_path / split
        d.mkdir(parents=True)
        (d / "_annotations.coco.json").write_text(
            json.dumps(
                {
                    "images": [
                        {"id": i, "file_name": f"{i}.jpg", "width": 8, "height": 8}
                        for i in range(n_images)
                    ],
                    "annotations": [
                        {
                            "id": j,
                            "image_id": 0,
                            "category_id": 1,
                            "segmentation": [[0, 0, 1, 0, 1, 1]],
                            "area": 1.0,
                            "bbox": [0, 0, 1, 1],
                            "iscrowd": 0,
                        }
                        for j in range(n_anns)
                    ],
                    "categories": [{"id": 1, "name": "ant"}],
                }
            )
        )
    return tmp_path


def test_role_has_a_geometry_level():
    assert role_min_level(TrainingRole.SEMANTIC_SAM3) is GeometryLevel.POLYGON


def test_validate_accepts_a_coco_layout(tmp_path):
    # validate_role_dataset used to call inspect_obb_or_detect_dataset
    # unconditionally, which RAISES on a COCO layout.
    report = validate_role_dataset(_coco(tmp_path), TrainingRole.SEMANTIC_SAM3)
    assert report.valid


def test_validate_rejects_an_empty_coco_dataset(tmp_path):
    # The old fall-through returned valid=True for unhandled roles, so a
    # forgotten validator would silently pass. It must actually inspect.
    bad = _coco(tmp_path, n_images=0, n_anns=0)
    report = validate_role_dataset(bad, TrainingRole.SEMANTIC_SAM3)
    assert not report.valid


def test_run_training_does_not_reach_the_ultralytics_builder(monkeypatch):
    from hydra_suite.training import runner

    called = {}
    monkeypatch.setattr(
        runner,
        "build_ultralytics_command",
        lambda *a, **k: called.setdefault("yolo", True),
    )
    monkeypatch.setattr(runner, "_train_sam3_lora", lambda *a, **k: {"success": True})
    from hydra_suite.training.contracts import (
        Sam3LoraParams,
        SourceDataset,
        TrainingHyperParams,
        TrainingRunSpec,
    )

    spec = TrainingRunSpec(
        role=TrainingRole.SEMANTIC_SAM3,
        source_datasets=[SourceDataset(path="/tmp/x", level="polygon")],
        derived_dataset_dir="/tmp/d",
        base_model="sam3",
        hyperparams=TrainingHyperParams(),
        sam3_params=Sam3LoraParams(prompt="ant"),
    )
    out = runner.run_training(spec, "/tmp/run")
    assert out["success"]
    assert "yolo" not in called
