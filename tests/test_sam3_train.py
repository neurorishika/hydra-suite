"""A training run that produces zero batches must report failure, never a
fake success -- see task-8 fix round 1, finding 2."""

from hydra_suite.training.contracts import (
    Sam3LoraParams,
    SourceDataset,
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)
from hydra_suite.training.sam3_lora import preflight as pf
from hydra_suite.training.sam3_lora import train as tr


def _healthy_spec(tmp_path):
    p = Sam3LoraParams(prompt="ant", label_quality_acknowledged=True)
    return TrainingRunSpec(
        role=TrainingRole.SEMANTIC_SAM3,
        source_datasets=[SourceDataset(path="/tmp/x", level="polygon")],
        derived_dataset_dir=str(tmp_path / "dataset"),
        base_model="sam3",
        hyperparams=TrainingHyperParams(),
        sam3_params=p,
    )


def test_zero_batches_reports_failure_not_success(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "_cuda_free_gb", lambda: 48.0)
    monkeypatch.setattr(pf, "_instance_count", lambda d: 100)
    monkeypatch.setattr(pf, "_free_disk_gb", lambda p: 100.0)
    monkeypatch.setattr(tr, "_build_dataloader", lambda spec, params, split: [])

    result = tr.train_sam3_lora(_healthy_spec(tmp_path), str(tmp_path / "run"))

    assert result["success"] is False
    assert result["artifact_path"] is None
    assert result["metrics_path"] is None
    assert "error_message" in result
    assert not (tmp_path / "run" / "adapters.pt").exists()
