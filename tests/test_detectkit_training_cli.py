"""Headless DetectKit training plan, workflow, and CLI tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace


def _plan_payload(tmp_path: Path) -> dict:
    return {
        "version": 1,
        "workspace": "./workspace",
        "sources": [{"path": "./source", "name": "day-1", "level": "obb"}],
        "class_names": ["ant"],
        "dataset": {
            "split": {"train": 0.8, "val": 0.2, "test": 0.0},
            "deduplicate": True,
            "crop_pad_ratio": 0.2,
            "min_crop_size_px": 96,
            "enforce_square": True,
            "slicing": {
                "enabled": True,
                "reference_body_px": 31.5,
                "target_size_fractions": [0.25, 0.5],
            },
        },
        "training": {
            "device": "0",
            "seed": 7,
            "epochs": 12,
            "batch": 4,
            "workers": 2,
            "augmentation": {"enabled": True, "args": {"fliplr": 0.5}},
        },
        "roles": [
            {"role": "detect_direct", "model": "yolo26s.pt", "imgsz": 768},
            {"role": "seq_detect", "model": "./models/stage1.pt", "imgsz": 640},
        ],
        "publish": {"auto_import": False, "auto_select": False},
    }


def test_plan_load_resolves_portable_paths_and_preserves_model_tokens(tmp_path):
    from hydra_suite.detectkit.config.training import load_training_plan

    config_path = tmp_path / "training.json"
    config_path.write_text(json.dumps(_plan_payload(tmp_path)), encoding="utf-8")

    plan = load_training_plan(config_path)

    assert plan.workspace_root == (tmp_path / "workspace").resolve()
    assert plan.sources[0].path == str((tmp_path / "source").resolve())
    assert plan.roles[0].base_model == "yolo26s.pt"
    assert plan.roles[1].base_model == str((tmp_path / "models/stage1.pt").resolve())
    assert plan.slice_settings.target_sizes_for(768) == [192.0, 384.0]
    assert plan.hyperparams.epochs == 12
    assert plan.publish_policy.auto_import is False


def test_plan_rejects_non_detectkit_roles_and_invalid_split(tmp_path):
    import pytest

    from hydra_suite.detectkit.config.training import (
        TrainingPlanError,
        load_training_plan,
    )

    payload = _plan_payload(tmp_path)
    payload["roles"] = [{"role": "classify_flat_yolo", "model": "model.pt"}]
    path = tmp_path / "bad-role.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TrainingPlanError, match="not a DetectKit training role"):
        load_training_plan(path)


def test_plan_rejects_string_booleans_and_boolean_numbers(tmp_path):
    import pytest

    from hydra_suite.detectkit.config.training import (
        TrainingPlanError,
        load_training_plan,
    )

    payload = _plan_payload(tmp_path)
    payload["dataset"]["deduplicate"] = "false"
    path = tmp_path / "string-bool.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TrainingPlanError, match="dataset.deduplicate.*boolean"):
        load_training_plan(path)

    payload = _plan_payload(tmp_path)
    payload["training"]["epochs"] = True
    path = tmp_path / "bool-number.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TrainingPlanError, match="training.epochs.*integer"):
        load_training_plan(path)


def test_plan_rejects_bad_nested_boolean_types(tmp_path):
    import pytest

    from hydra_suite.detectkit.config.training import (
        TrainingPlanError,
        load_training_plan,
    )

    payload = _plan_payload(tmp_path)
    payload["dataset"]["slicing"]["enabled"] = "false"
    path = tmp_path / "bad-slicing.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TrainingPlanError, match="dataset.slicing.enabled.*boolean"):
        load_training_plan(path)

    payload = _plan_payload(tmp_path)
    payload["publish"]["auto_import"] = "false"
    path = tmp_path / "bad-publish.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TrainingPlanError, match="publish.auto_import.*boolean"):
        load_training_plan(path)


def test_plan_rejects_nonfinite_numbers_and_empty_sam3_prompt(tmp_path):
    import pytest

    from hydra_suite.detectkit.config.training import (
        TrainingPlanError,
        load_training_plan,
    )

    payload = _plan_payload(tmp_path)
    payload["training"]["lr0"] = float("nan")
    path = tmp_path / "nan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TrainingPlanError, match="training.lr0.*finite"):
        load_training_plan(path)

    payload = _plan_payload(tmp_path)
    payload["roles"] = [{"role": "semantic_sam3", "imgsz": 1008}]
    payload["sam3"] = {"prompt": "   ", "label_quality_acknowledged": True}
    path = tmp_path / "empty-prompt.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        TrainingPlanError, match="SAM3 training requires a non-empty prompt"
    ):
        load_training_plan(path)

    payload = _plan_payload(tmp_path)
    payload["dataset"]["split"] = {"train": 0.9, "val": 0.2}
    path = tmp_path / "bad-split.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TrainingPlanError, match="sum to 1.0"):
        load_training_plan(path)


def test_headless_plan_defaults_to_no_server_local_publish(tmp_path):
    from hydra_suite.detectkit.config.training import DetectTrainingPlan

    payload = _plan_payload(tmp_path)
    payload.pop("publish")

    plan = DetectTrainingPlan.from_dict(payload, base_dir=tmp_path)

    assert plan.publish_policy.auto_import is False
    assert plan.publish_policy.auto_select is False


def test_shared_workflow_prepares_and_runs_roles_with_parent_lineage(tmp_path):
    from hydra_suite.detectkit.config.training import DetectTrainingPlan
    from hydra_suite.detectkit.jobs.training import (
        prepare_role_datasets,
        run_role_entries,
    )

    plan = DetectTrainingPlan.from_dict(_plan_payload(tmp_path), base_dir=tmp_path)
    calls: dict[str, list] = {"merge": [], "slice": [], "role": [], "run": []}

    class _Orchestrator:
        def build_merged_obb_dataset(self, *_args, **kwargs):
            calls["merge"].append(kwargs)
            return SimpleNamespace(
                dataset_dir="/tmp/merged",
                stats={"source_items": {"source": 1}},
            )

        def build_sliced_obb_dataset(self, source_dir, **kwargs):
            calls["slice"].append((source_dir, kwargs))
            return SimpleNamespace(
                dataset_dir="/tmp/sliced",
                stats={"measured_reference_body_px": 42.0},
            )

        def build_role_dataset(self, role, source_dir, **kwargs):
            calls["role"].append((role.value, source_dir, kwargs))
            return SimpleNamespace(dataset_dir=f"/tmp/{role.value}")

        def run_role_training(self, spec, **kwargs):
            calls["run"].append((spec, kwargs))
            return {"success": True, "run_id": f"run-{spec.role.value}"}

    orchestrator = _Orchestrator()
    prepared = prepare_role_datasets(
        orchestrator,
        plan.preparation_request(),
        log=lambda _message: None,
        status=lambda _message: None,
        should_cancel=lambda: False,
    )
    entries = plan.role_entries(prepared.role_dataset_dirs)
    results = run_role_entries(
        orchestrator,
        entries,
        log=lambda _message: None,
        progress=lambda _role, _current, _total: None,
        should_cancel=lambda: False,
    )

    assert len(calls["merge"]) == 1
    assert len(calls["slice"]) == 1
    assert calls["slice"][0][1]["params"].reference_body_px == 31.5
    assert [source for _, source, _ in calls["role"]] == [
        "/tmp/sliced",
        "/tmp/merged",
    ]
    assert calls["run"][0][1]["parent_run_id"] == ""
    assert calls["run"][1][1]["parent_run_id"] == "run-detect_direct"
    assert calls["run"][0][0].hyperparams.imgsz == 768
    assert calls["run"][0][0].publish_policy.auto_import is False
    assert [result["role"] for result in results] == [
        "detect_direct",
        "seq_detect",
    ]


def test_preflight_uses_native_polygon_geometry(tmp_path):
    from hydra_suite.detectkit.jobs.training import preflight_sources
    from hydra_suite.training import SourceDataset

    source = tmp_path / "polygon-source"
    (source / "images/train").mkdir(parents=True)
    (source / "labels/train").mkdir(parents=True)
    (source / "images/train/frame.png").write_bytes(b"image-placeholder")
    (source / "labels/train/frame.txt").write_text(
        "0 0.1 0.1 0.4 0.1 0.5 0.3 0.3 0.5 0.1 0.4\n",
        encoding="utf-8",
    )

    report = preflight_sources((SourceDataset(path=str(source), level="polygon"),))

    assert report.valid is True
    assert report.stats["sources"][0]["level"] == "polygon"


def test_cli_dry_run_is_qt_free_and_does_not_create_workspace(tmp_path, monkeypatch):
    from hydra_suite.detectkit import cli

    payload = _plan_payload(tmp_path)
    payload["workspace"] = "./must-not-exist"
    config_path = tmp_path / "training.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "PySide6", None)

    assert cli.main(["--config", str(config_path), "--dry-run"]) == 0
    assert not (tmp_path / "must-not-exist").exists()


def test_detectkit_app_dispatches_train_without_importing_qt(monkeypatch):
    from hydra_suite.detectkit import app

    received = []
    fake_cli = SimpleNamespace(main=lambda argv: received.append(argv) or 17)
    monkeypatch.setitem(sys.modules, "hydra_suite.detectkit.cli", fake_cli)
    monkeypatch.setitem(sys.modules, "PySide6", None)

    assert app.main(["train", "--config", "run.json"]) == 17
    assert received == [["--config", "run.json"]]


def test_detectkit_train_dry_run_blocks_all_pyside_imports_in_subprocess(tmp_path):
    payload = _plan_payload(tmp_path)
    config_path = tmp_path / "training.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    script = """
import builtins
import sys
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'PySide6' or name.startswith('PySide6.'):
        raise AssertionError('headless training imported Qt: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from hydra_suite.detectkit.app import main
raise SystemExit(main(['train', '--config', sys.argv[1], '--dry-run']))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    completed = subprocess.run(
        [sys.executable, "-c", script, str(config_path)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"detect_direct"' in completed.stdout


def test_resume_rewrites_single_role_spec(tmp_path):
    import pytest

    from hydra_suite.detectkit.cli import _apply_resume
    from hydra_suite.detectkit.config.training import (
        DetectTrainingPlan,
        TrainingPlanError,
    )

    payload = _plan_payload(tmp_path)
    payload["roles"] = payload["roles"][:1]
    plan = DetectTrainingPlan.from_dict(payload, base_dir=tmp_path)
    entries = plan.role_entries({"detect_direct": "/tmp/detect_direct"})
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"checkpoint")

    resumed = _apply_resume(entries, "last.pt", tmp_path)

    assert resumed[0].spec.base_model == str(checkpoint)
    assert resumed[0].spec.resume_from == str(checkpoint)

    with pytest.raises(TrainingPlanError, match="exactly one role"):
        _apply_resume(entries * 2, "last.pt", tmp_path)


def test_cli_rejects_invalid_resume_before_dataset_preparation(tmp_path, monkeypatch):
    from hydra_suite.detectkit import cli

    payload = _plan_payload(tmp_path)
    config_path = tmp_path / "training.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    called = False

    def prepare(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("dataset preparation should not run")

    monkeypatch.setattr(cli, "prepare_role_datasets", prepare)

    assert cli.main(["--config", str(config_path), "--resume", "missing.pt"]) == 2
    assert called is False
    assert not (tmp_path / "workspace").exists()


def test_repeated_cancel_signal_only_requests_cooperative_shutdown(monkeypatch):
    from hydra_suite.detectkit import cli

    installed = {}
    monkeypatch.setattr(cli.signal, "getsignal", lambda _signum: object())
    monkeypatch.setattr(
        cli.signal,
        "signal",
        lambda signum, handler: installed.update({signum: handler}),
    )
    event = threading.Event()

    cli._install_cancel_handlers(event)
    handler = installed[cli.signal.SIGTERM]
    handler(cli.signal.SIGTERM, None)
    handler(cli.signal.SIGTERM, None)

    assert event.is_set()


def test_workspace_session_is_unique_and_refuses_concurrent_use(tmp_path):
    import pytest

    from hydra_suite.detectkit.cli import _workspace_session
    from hydra_suite.detectkit.config.training import TrainingPlanError

    workspace = tmp_path / "workspace"
    with _workspace_session(workspace) as first:
        with pytest.raises(TrainingPlanError, match="already in use"):
            with _workspace_session(workspace):
                pass

    with _workspace_session(workspace) as second:
        pass

    assert first != second
    assert first.parent == workspace / "sessions"
    assert second.parent == workspace / "sessions"


def test_cli_writes_durable_success_summary(tmp_path, monkeypatch):
    from hydra_suite.detectkit import cli
    from hydra_suite.detectkit.jobs.training import DatasetPreparationResult
    from hydra_suite.training import TrainingRole, ValidationReport

    payload = _plan_payload(tmp_path)
    payload["roles"] = payload["roles"][:1]
    config_path = tmp_path / "training.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(cli, "TrainingOrchestrator", lambda _workspace: object())
    monkeypatch.setattr(
        cli,
        "preflight_sources",
        lambda _sources: ValidationReport(valid=True),
    )
    monkeypatch.setattr(
        cli,
        "prepare_role_datasets",
        lambda *_args, **_kwargs: DatasetPreparationResult(
            role_dataset_dirs={"detect_direct": "/tmp/detect"},
            roles=(TrainingRole.DETECT_DIRECT,),
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_role_entries",
        lambda *_args, **_kwargs: [
            {
                "role": "detect_direct",
                "run_id": "run-1",
                "success": True,
                "artifact_path": Path("/tmp/best.pt"),
            }
        ],
    )

    assert cli.main(["--config", str(config_path)]) == 0
    session_dirs = list((tmp_path / "workspace/sessions").iterdir())
    assert len(session_dirs) == 1
    summary = json.loads(
        (session_dirs[0] / "training_result.json").read_text(encoding="utf-8")
    )
    assert summary["success"] is True
    assert summary["results"][0]["artifact_path"] == "/tmp/best.pt"


def test_cli_returns_configuration_error_for_invalid_json(tmp_path, capsys):
    from hydra_suite.detectkit import cli

    config_path = tmp_path / "broken.json"
    config_path.write_text("{not-json", encoding="utf-8")

    assert cli.main(["--config", str(config_path)]) == 2
    assert "Invalid JSON" in capsys.readouterr().err
