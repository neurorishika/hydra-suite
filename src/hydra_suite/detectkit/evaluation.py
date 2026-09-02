"""Validation-set evaluation contracts and services for DetectKit models."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from hydra_suite.detectkit.gui.models import DetectKitProject


_ROLE_TASKS = {
    "obb_direct": "obb",
    "detect_direct": "detect",
    "segment_direct": "segment",
    "seq_detect": "detect",
    "seq_crop_obb": "obb",
    "seq_crop_segment": "segment",
}

_ROLE_IMGSZ_FIELDS = {
    "obb_direct": "imgsz_obb_direct",
    "detect_direct": "imgsz_detect_direct",
    "segment_direct": "imgsz_segment_direct",
    "seq_detect": "imgsz_seq_detect",
    "seq_crop_obb": "imgsz_seq_crop_obb",
    "seq_crop_segment": "imgsz_seq_crop_segment",
}


@dataclass(frozen=True, slots=True)
class EvaluationCandidate:
    """One training run and the held-out dataset required to evaluate it."""

    run_id: str
    role: str
    task: str
    model_path: str
    dataset_dir: str
    dataset_yaml: str
    imgsz: int
    available: bool
    unavailable_reason: str = ""


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Task-level validation metrics returned by an Ultralytics validation run."""

    evaluation_id: str
    run_id: str
    role: str
    model_path: str
    dataset_dir: str
    precision: float = 0.0
    recall: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0
    elapsed_seconds: float = 0.0
    inference_ms: float = 0.0
    evaluated_at: str = ""
    error: str = ""

    @property
    def success(self) -> bool:
        return not bool(self.error)

    @classmethod
    def failed(
        cls,
        candidate: EvaluationCandidate,
        evaluation_id: str,
        message: str,
    ) -> "EvaluationResult":
        return cls(
            evaluation_id=evaluation_id,
            run_id=candidate.run_id,
            role=candidate.role,
            model_path=candidate.model_path,
            dataset_dir=candidate.dataset_dir,
            evaluated_at=datetime.now().isoformat(timespec="seconds"),
            error=str(message),
        )

    def to_project_record(self) -> dict[str, Any]:
        """Return portable history data without external model or dataset paths."""
        record = asdict(self)
        record["model_name"] = Path(self.model_path).name
        record["dataset_name"] = Path(self.dataset_dir).name
        record.pop("model_path", None)
        record.pop("dataset_dir", None)
        return record


def _entry_model_path(entry: dict[str, Any]) -> str:
    paths: list[str] = []
    for key in ("project_model_paths",):
        for path in entry.get(key) or []:
            candidate = str(path or "").strip()
            if candidate:
                paths.append(candidate)
    for key in ("project_model_path", "published_model_path"):
        candidate = str(entry.get(key, "") or "").strip()
        if candidate:
            paths.append(candidate)
    for path in entry.get("artifact_paths") or []:
        candidate = str(path or "").strip()
        if candidate:
            paths.append(candidate)
    return next(
        (path for path in paths if Path(path).expanduser().is_file()),
        paths[0] if paths else "",
    )


def _dataset_paths(entry: dict[str, Any]) -> tuple[str, str]:
    spec = entry.get("spec") or {}
    raw_ref = str(spec.get("derived_dataset_dir", "") or "").strip()
    if not raw_ref:
        return "", ""
    dataset_ref = Path(raw_ref).expanduser()
    if dataset_ref.suffix.lower() in {".yaml", ".yml"}:
        return str(dataset_ref.parent), str(dataset_ref)
    return str(dataset_ref), str(dataset_ref / "dataset.yaml")


def _candidate_imgsz(
    project: "DetectKitProject", entry: dict[str, Any], role: str
) -> int:
    spec = entry.get("spec") or {}
    hyperparams = spec.get("hyperparams") or {}
    try:
        value = int(hyperparams.get("imgsz", 0) or 0)
    except (TypeError, ValueError):
        value = 0
    if value > 0:
        return value
    field_name = _ROLE_IMGSZ_FIELDS.get(role, "imgsz_obb_direct")
    return max(1, int(getattr(project, field_name, 640) or 640))


def collect_evaluation_candidates(
    project: "DetectKitProject",
) -> list[EvaluationCandidate]:
    """Resolve training-history runs into transparent evaluation choices."""
    candidates: list[EvaluationCandidate] = []
    for entry in reversed(list(project.training_history or [])):
        run_id = str(entry.get("run_id", "") or "unknown")
        role = str(entry.get("role", "") or "").strip().lower()
        task = _ROLE_TASKS.get(role, "")
        model_path = _entry_model_path(entry)
        dataset_dir, dataset_yaml = _dataset_paths(entry)
        reason = ""

        status = str(entry.get("status", "") or "").strip().lower()
        if status and status != "completed":
            reason = f"Run status is {status}; only completed runs can be evaluated."
        elif not task:
            reason = f"Validation is not supported for role '{role or 'unknown'}'."
        elif not model_path or not Path(model_path).expanduser().is_file():
            reason = "The trained model artifact is missing."
        elif not dataset_yaml or not Path(dataset_yaml).expanduser().is_file():
            reason = "The derived dataset or its dataset.yaml is missing."

        candidates.append(
            EvaluationCandidate(
                run_id=run_id,
                role=role,
                task=task,
                model_path=model_path,
                dataset_dir=dataset_dir,
                dataset_yaml=dataset_yaml,
                imgsz=_candidate_imgsz(project, entry, role),
                available=not bool(reason),
                unavailable_reason=reason,
            )
        )
    return candidates


def _load_model(model_path: str, task: str):
    from ultralytics import YOLO

    return YOLO(model_path, task=task)


def _metric_value(results: dict[str, Any], name: str, suffix: str) -> float:
    key = f"metrics/{name}({suffix})"
    if key not in results:
        raise RuntimeError(f"Ultralytics validation did not return '{key}'.")
    value = float(results[key])
    if not math.isfinite(value):
        raise RuntimeError(
            f"Ultralytics validation returned a non-finite value for '{key}'. "
            "Check that the val split contains labeled instances."
        )
    return value


def new_evaluation_id(run_id: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_run = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in run_id)
    return f"{timestamp}-{safe_run[:48]}-{uuid4().hex[:8]}"


def evaluate_candidate(
    candidate: EvaluationCandidate,
    *,
    output_root: str | Path,
    device: str,
    batch: int,
    evaluation_id: str | None = None,
) -> EvaluationResult:
    """Evaluate one model on its held-out ``val`` split and persist metrics."""
    if not candidate.available:
        raise ValueError(
            candidate.unavailable_reason or "Evaluation run is unavailable."
        )

    evaluation_id = evaluation_id or new_evaluation_id(candidate.run_id)
    output_root = Path(output_root).expanduser().resolve()
    evaluation_dir = output_root / evaluation_id
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(candidate.model_path, candidate.task)
    val_args: dict[str, Any] = {
        "data": candidate.dataset_yaml,
        "split": "val",
        "imgsz": int(candidate.imgsz),
        "batch": max(1, int(batch)),
        "workers": 0,
        "plots": False,
        "save_json": False,
        "verbose": False,
        "project": str(output_root),
        "name": evaluation_id,
        "exist_ok": True,
    }
    selected_device = str(device or "").strip()
    if selected_device and selected_device.lower() != "auto":
        val_args["device"] = selected_device

    started = time.perf_counter()
    metrics = model.val(**val_args)
    elapsed = time.perf_counter() - started
    results_dict = dict(getattr(metrics, "results_dict", {}) or {})
    suffix = "M" if candidate.task == "segment" else "B"
    speed = dict(getattr(metrics, "speed", {}) or {})
    result = EvaluationResult(
        evaluation_id=evaluation_id,
        run_id=candidate.run_id,
        role=candidate.role,
        model_path=candidate.model_path,
        dataset_dir=candidate.dataset_dir,
        precision=_metric_value(results_dict, "precision", suffix),
        recall=_metric_value(results_dict, "recall", suffix),
        map50=_metric_value(results_dict, "mAP50", suffix),
        map50_95=_metric_value(results_dict, "mAP50-95", suffix),
        elapsed_seconds=float(elapsed),
        inference_ms=float(speed.get("inference", 0.0) or 0.0),
        evaluated_at=datetime.now().isoformat(timespec="seconds"),
    )

    metrics_path = evaluation_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
