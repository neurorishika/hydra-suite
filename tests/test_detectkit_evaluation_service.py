"""Tests for DetectKit validation-set evaluation services."""

from __future__ import annotations

import json
from pathlib import Path


def _history_entry(
    tmp_path: Path,
    *,
    run_id: str = "run_001",
    role: str = "obb_direct",
) -> tuple[dict, Path, Path]:
    model_path = tmp_path / "models" / f"{run_id}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"weights")
    dataset_dir = tmp_path / "derived" / run_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames: [ant]\n",
        encoding="utf-8",
    )
    entry = {
        "run_id": run_id,
        "role": role,
        "status": "completed",
        "success": True,
        "project_model_path": str(model_path),
        "spec": {
            "derived_dataset_dir": str(dataset_dir),
            "hyperparams": {"imgsz": 768},
        },
    }
    return entry, model_path, dataset_dir


def test_candidates_resolve_model_and_validation_dataset(tmp_path):
    from hydra_suite.detectkit.evaluation import collect_evaluation_candidates
    from hydra_suite.detectkit.gui.models import DetectKitProject

    entry, model_path, dataset_dir = _history_entry(tmp_path)
    project = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    project.training_history = [entry]

    candidates = collect_evaluation_candidates(project)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.available is True
    assert candidate.model_path == str(model_path)
    assert candidate.dataset_yaml == str(dataset_dir / "dataset.yaml")
    assert candidate.imgsz == 768
    assert candidate.task == "obb"


def test_candidates_explain_unsupported_or_missing_runs(tmp_path):
    from hydra_suite.detectkit.evaluation import collect_evaluation_candidates
    from hydra_suite.detectkit.gui.models import DetectKitProject

    sam_entry, _, _ = _history_entry(
        tmp_path,
        run_id="sam_run",
        role="semantic_sam3",
    )
    missing_entry, missing_model, _ = _history_entry(
        tmp_path,
        run_id="missing_model",
        role="detect_direct",
    )
    missing_model.unlink()
    project = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    project.training_history = [sam_entry, missing_entry]

    candidates = collect_evaluation_candidates(project)

    assert len(candidates) == 2
    reasons = {
        candidate.run_id: candidate.unavailable_reason for candidate in candidates
    }
    assert "not supported" in reasons["sam_run"].lower()
    assert "model" in reasons["missing_model"].lower()


def test_evaluate_candidate_runs_ultralytics_val_and_extracts_box_metrics(
    tmp_path, monkeypatch
):
    from hydra_suite.detectkit import evaluation as evaluation_service
    from hydra_suite.detectkit.gui.models import DetectKitProject

    entry, _, _ = _history_entry(tmp_path)
    project = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    project.training_history = [entry]
    candidate = evaluation_service.collect_evaluation_candidates(project)[0]
    captured = {}

    class _Metrics:
        results_dict = {
            "metrics/precision(B)": 0.81,
            "metrics/recall(B)": 0.72,
            "metrics/mAP50(B)": 0.76,
            "metrics/mAP50-95(B)": 0.51,
        }
        speed = {"inference": 4.2}

    class _Model:
        def val(self, **kwargs):
            captured["val"] = kwargs
            return _Metrics()

    def _load_model(model_path, task):
        captured["model_path"] = model_path
        captured["task"] = task
        return _Model()

    monkeypatch.setattr(evaluation_service, "_load_model", _load_model)

    result = evaluation_service.evaluate_candidate(
        candidate,
        output_root=tmp_path / "evaluation",
        device="cpu",
        batch=4,
        evaluation_id="eval-001",
    )

    assert result.success is True
    assert result.precision == 0.81
    assert result.recall == 0.72
    assert result.map50 == 0.76
    assert result.map50_95 == 0.51
    assert captured["task"] == "obb"
    assert captured["val"]["split"] == "val"
    assert captured["val"]["data"] == candidate.dataset_yaml
    assert captured["val"]["imgsz"] == 768
    assert captured["val"]["batch"] == 4
    metrics_file = tmp_path / "evaluation" / "eval-001" / "metrics.json"
    assert json.loads(metrics_file.read_text(encoding="utf-8"))["map50"] == 0.76


def test_segment_evaluation_reports_mask_metrics_not_box_metrics(tmp_path, monkeypatch):
    from hydra_suite.detectkit import evaluation as evaluation_service
    from hydra_suite.detectkit.gui.models import DetectKitProject

    entry, _, _ = _history_entry(tmp_path, role="segment_direct")
    project = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    project.training_history = [entry]
    candidate = evaluation_service.collect_evaluation_candidates(project)[0]

    class _Metrics:
        results_dict = {
            "metrics/precision(B)": 0.91,
            "metrics/recall(B)": 0.92,
            "metrics/mAP50(B)": 0.93,
            "metrics/mAP50-95(B)": 0.94,
            "metrics/precision(M)": 0.61,
            "metrics/recall(M)": 0.62,
            "metrics/mAP50(M)": 0.63,
            "metrics/mAP50-95(M)": 0.64,
        }
        speed = {}

    class _Model:
        def val(self, **_kwargs):
            return _Metrics()

    monkeypatch.setattr(
        evaluation_service,
        "_load_model",
        lambda _model_path, _task: _Model(),
    )

    result = evaluation_service.evaluate_candidate(
        candidate,
        output_root=tmp_path / "evaluation",
        device="cpu",
        batch=1,
        evaluation_id="eval-segment",
    )

    assert result.precision == 0.61
    assert result.recall == 0.62
    assert result.map50 == 0.63
    assert result.map50_95 == 0.64


def test_evaluation_result_project_record_omits_external_paths():
    from hydra_suite.detectkit.evaluation import EvaluationResult

    result = EvaluationResult(
        evaluation_id="eval-001",
        run_id="run-001",
        role="obb_direct",
        model_path="/external/models/best.pt",
        dataset_dir="/external/datasets/derived",
        precision=0.8,
        recall=0.7,
        map50=0.75,
        map50_95=0.5,
        elapsed_seconds=3.0,
        inference_ms=2.0,
        evaluated_at="2026-09-02T12:00:00",
    )

    record = result.to_project_record()

    assert record["model_name"] == "best.pt"
    assert record["dataset_name"] == "derived"
    assert "model_path" not in record
    assert "dataset_dir" not in record
