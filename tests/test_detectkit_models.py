"""Tests for DetectKitProject model — active_model_path field."""

from __future__ import annotations


def test_project_active_model_path_persists(tmp_path):
    from hydra_suite.detectkit.gui.models import DetectKitProject

    proj = DetectKitProject(project_dir=tmp_path)
    proj.active_model_path = "/some/model.pt"
    save_path = tmp_path / "project.json"
    proj.save(save_path)

    loaded = DetectKitProject.load(save_path)
    assert loaded.active_model_path == "/some/model.pt"


def test_project_load_missing_active_model_path_defaults(tmp_path):
    """Old project files without active_model_path should load without error."""
    import json

    from hydra_suite.detectkit.gui.models import DetectKitProject

    # Write a minimal project JSON without active_model_path
    data = {"version": 1, "class_names": ["ant"]}
    save_path = tmp_path / "project.json"
    save_path.write_text(json.dumps(data), encoding="utf-8")

    proj = DetectKitProject.load(save_path)
    assert proj.active_model_path == ""


def test_project_load_falls_back_from_unavailable_cuda_to_mps(tmp_path, monkeypatch):
    """Projects moved to Apple Silicon must not retain an unusable CUDA setting."""
    import json

    from hydra_suite.core.inference import torch_device
    from hydra_suite.detectkit.gui.models import DetectKitProject

    monkeypatch.setattr(torch_device, "TORCH_CUDA_AVAILABLE", False)
    monkeypatch.setattr(torch_device, "MPS_AVAILABLE", True)
    save_path = tmp_path / "project.json"
    save_path.write_text(json.dumps({"version": 1, "device": "cuda"}), encoding="utf-8")

    assert DetectKitProject.load(save_path).device == "mps"


def test_project_load_preserves_explicit_cpu_device(tmp_path, monkeypatch):
    import json

    from hydra_suite.core.inference import torch_device
    from hydra_suite.detectkit.gui.models import DetectKitProject

    monkeypatch.setattr(torch_device, "TORCH_CUDA_AVAILABLE", True)
    monkeypatch.setattr(torch_device, "MPS_AVAILABLE", True)
    save_path = tmp_path / "project.json"
    save_path.write_text(json.dumps({"version": 1, "device": "cpu"}), encoding="utf-8")

    assert DetectKitProject.load(save_path).device == "cpu"


def test_project_to_dict_includes_active_model_path():
    from hydra_suite.detectkit.gui.models import DetectKitProject

    proj = DetectKitProject()
    proj.active_model_path = "weights/best.pt"
    d = proj.to_dict()
    assert "active_model_path" in d
    assert d["active_model_path"] == "weights/best.pt"
    assert d["active_model_path"] == "weights/best.pt"


def test_project_training_history_persists(tmp_path):
    from hydra_suite.detectkit.gui.models import DetectKitProject

    proj = DetectKitProject(project_dir=tmp_path)
    proj.training_history = [{"run_id": "run_001", "project_model_path": "model.pt"}]
    save_path = tmp_path / "project.json"
    proj.save(save_path)

    loaded = DetectKitProject.load(save_path)
    assert loaded.training_history == [
        {"run_id": "run_001", "project_model_path": "model.pt"}
    ]


def test_project_evaluation_history_persists(tmp_path):
    from hydra_suite.detectkit.gui.models import DetectKitProject

    record = {
        "evaluation_id": "eval_001",
        "run_id": "run_001",
        "precision": 0.8,
        "recall": 0.7,
        "map50": 0.75,
        "map50_95": 0.5,
    }
    proj = DetectKitProject(project_dir=tmp_path)
    proj.evaluation_history = [record]
    save_path = tmp_path / "project.json"
    proj.save(save_path)

    loaded = DetectKitProject.load(save_path)

    assert loaded.evaluation_history == [record]


from hydra_suite.detectkit.gui.models import OBBSource, StagedReview


def test_staged_review_round_trips():
    review = StagedReview(
        staged_path="/tmp/staging",
        target_level="polygon",
        producer="sam3",
        producer_variant="sam3-large",
        prompt="ant",
        params={"confidence": 0.35},
        created_at="2026-08-31T10:00:00",
    )

    restored = StagedReview.from_dict(review.to_dict())

    assert restored == review


def test_to_dict_writes_only_the_new_key_names():
    d = StagedReview(producer="sam2", producer_variant="sam2.1_hiera_large").to_dict()

    assert set(d) == {
        "staged_path",
        "target_level",
        "producer",
        "producer_variant",
        "prompt",
        "params",
        "created_at",
    }


def test_from_dict_accepts_a_legacy_sam2_record():
    legacy = {
        "staged_path": "/tmp/s",
        "target_level": "polygon",
        "sam2_variant": "sam2.1_hiera_large",
        "created_at": "2026-08-01T00:00:00",
    }

    review = StagedReview.from_dict(legacy)

    assert review.producer == "sam2"
    assert review.producer_variant == "sam2.1_hiera_large"
    assert review.prompt == ""


def test_from_dict_accepts_a_legacy_sam3_record():
    legacy = {
        "staged_path": "/tmp/s",
        "target_level": "polygon",
        "primer_kind": "sam3",
        "primer_variant": "sam3-large",
        "primer_prompt": "ant",
        "primer_params": {"confidence": 0.35},
        "created_at": "2026-08-01T00:00:00",
    }

    review = StagedReview.from_dict(legacy)

    assert review.producer == "sam3"
    assert review.producer_variant == "sam3-large"
    assert review.prompt == "ant"
    assert review.params == {"confidence": 0.35}


def test_source_loads_a_legacy_pending_escalation_key():
    src = OBBSource.from_dict(
        {
            "path": "/tmp/src",
            "name": "src",
            "pending_escalation": {
                "staged_path": "/tmp/s",
                "target_level": "polygon",
                "primer_kind": "sam3",
                "primer_prompt": "ant",
            },
        }
    )

    assert src.staged_review is not None
    assert src.staged_review.producer == "sam3"


def test_source_writes_the_new_key_name():
    src = OBBSource(path="/tmp/src", name="src", staged_review=StagedReview())

    d = src.to_dict()

    assert "staged_review" in d
    assert "pending_escalation" not in d
