"""SEMANTIC_SAM3 must fork to publish_sam3_model, never publish_trained_model."""

import pytest

import hydra_suite.training.service as svc
from hydra_suite.training.contracts import (
    PublishPolicy,
    Sam3LoraParams,
    SourceDataset,
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)
from hydra_suite.training.registry import load_registry


def _spec(role=TrainingRole.SEMANTIC_SAM3):
    return TrainingRunSpec(
        role=role,
        source_datasets=[SourceDataset(path="/tmp/x", level="polygon")],
        derived_dataset_dir="/tmp/d",
        base_model="sam3",
        hyperparams=TrainingHyperParams(),
        sam3_params=Sam3LoraParams(prompt="ant"),
    )


def test_sam3_role_forks_to_publish_sam3_model(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(
        svc, "publish_trained_model", lambda *a, **k: called.setdefault("yolo", True)
    )
    monkeypatch.setattr(
        svc,
        "publish_sam3_model",
        lambda **k: (called.setdefault("sam3", True), ("key", "/tmp/a.pt"))[1],
    )
    monkeypatch.setattr(svc, "ensure_checkpoint", lambda *a, **k: "/tmp/base.pt")
    key, path = svc._publish_training_artifacts(
        spec=_spec(),
        artifact_paths=[str(tmp_path / "adapters.pt")],
        publish_metadata={},
        run_id="r1",
        dataset_fingerprint_value="fp1",
    )
    assert "sam3" in called
    # publish_trained_model raises "Unsupported publish role" for this role.
    assert "yolo" not in called
    assert path == "/tmp/a.pt"


def test_other_roles_still_use_the_yolo_publisher(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(
        svc,
        "publish_trained_model",
        lambda *a, **k: (called.setdefault("yolo", True), ("k", "/tmp/y.pt"))[1],
    )
    monkeypatch.setattr(
        svc, "publish_sam3_model", lambda **k: called.setdefault("sam3", True)
    )
    svc._publish_training_artifacts(
        spec=_spec(TrainingRole.SEGMENT_DIRECT),
        artifact_paths=[str(tmp_path / "best.pt")],
        publish_metadata={},
        run_id="r1",
        dataset_fingerprint_value="fp1",
    )
    assert "yolo" in called and "sam3" not in called


def test_builder_geometry_reaches_the_sidecar(monkeypatch, tmp_path):
    import json

    derived = tmp_path / "derived"
    derived.mkdir()
    (derived / "build_manifest.json").write_text(
        json.dumps({"tile_px": 1007, "reference_body_px": 55.4})
    )
    seen = {}
    monkeypatch.setattr(
        svc, "publish_sam3_model", lambda **k: (seen.update(k), ("key", "/tmp/a.pt"))[1]
    )
    monkeypatch.setattr(svc, "ensure_checkpoint", lambda *a, **k: "/tmp/base.pt")
    spec = _spec()
    spec.derived_dataset_dir = str(derived)
    svc._publish_training_artifacts(
        spec=spec,
        artifact_paths=[str(tmp_path / "adapters.pt")],
        publish_metadata={"species": "ant"},
        run_id="r1",
        dataset_fingerprint_value="fp1",
    )
    assert seen["build_manifest"]["tile_px"] == 1007
    assert seen["source_fingerprint"] == "fp1"


def _registered_spec(tmp_path, *, auto_import=False):
    derived = tmp_path / "derived"
    derived.mkdir(exist_ok=True)
    return TrainingRunSpec(
        role=TrainingRole.SEMANTIC_SAM3,
        source_datasets=[SourceDataset(path=str(derived), level="polygon")],
        derived_dataset_dir=str(derived),
        base_model="sam3",
        hyperparams=TrainingHyperParams(),
        publish_policy=PublishPolicy(auto_import=auto_import),
        sam3_params=Sam3LoraParams(prompt="ant"),
    )


def test_direct_orchestrator_rejects_oversized_prompt_pool_before_registry_work(
    monkeypatch, tmp_path
):
    from hydra_suite.training.contracts import SAM3_MAX_NEGATIVE_PROMPT_COUNT

    class BombList(list):
        def __iter__(self):
            pytest.fail("over-cardinality prompt list must not be iterated")

    spec = _registered_spec(tmp_path)
    spec.sam3_params.negative_prompts = BombList(
        ["x"] * (SAM3_MAX_NEGATIVE_PROMPT_COUNT + 1)
    )
    monkeypatch.setattr(
        svc,
        "dataset_fingerprint",
        lambda *_args: pytest.fail("fingerprint must follow SAM3 text admission"),
    )
    monkeypatch.setattr(
        svc,
        "create_run_record",
        lambda *_args, **_kwargs: pytest.fail("registry must follow text admission"),
    )

    with pytest.raises(ValueError, match="4096 entries"):
        svc.TrainingOrchestrator(tmp_path / "workspace").run_role_training(spec)


@pytest.mark.parametrize(
    "prompt,negative_prompts,match",
    [
        (123, [], "prompt must be a string"),
        ("ant", "background", "must be a list or tuple"),
        ("ant", ["x" * 256] * 1025, "serialized text cap"),
    ],
)
def test_direct_orchestrator_validates_prompt_shape_before_fingerprint(
    monkeypatch, tmp_path, prompt, negative_prompts, match
):
    spec = _registered_spec(tmp_path)
    spec.sam3_params.prompt = prompt
    spec.sam3_params.negative_prompts = negative_prompts
    monkeypatch.setattr(
        svc,
        "dataset_fingerprint",
        lambda *_args: pytest.fail("fingerprint must follow SAM3 text admission"),
    )

    with pytest.raises(ValueError, match=match):
        svc.TrainingOrchestrator(tmp_path / "workspace").run_role_training(spec)


def test_training_exception_finalizes_registry_and_preserves_run_identity(
    monkeypatch, tmp_path
):
    import hydra_suite.training.registry as registry

    monkeypatch.setattr(registry, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        svc,
        "run_training",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("sidecar bootstrap exploded")
        ),
    )

    result = svc.TrainingOrchestrator(tmp_path / "workspace").run_role_training(
        _registered_spec(tmp_path)
    )

    assert result["success"] is False
    assert result["run_id"]
    assert result["failure_kind"] == "training-exception"
    assert result["error"] == "sidecar bootstrap exploded"
    record = load_registry()["runs"][0]
    assert record["run_id"] == result["run_id"]
    assert record["status"] == "failed"
    assert record["finished_at"]
    assert record["failure_kind"] == "training-exception"
    assert record["error_message"] == "sidecar bootstrap exploded"


def test_owned_workload_error_is_reraised_with_recovery_handle(monkeypatch, tmp_path):
    import hydra_suite.training.registry as registry
    from hydra_suite.runtime.process_supervisor import WorkloadStillOwnedError

    monkeypatch.setattr(registry, "_project_root", lambda: tmp_path)
    recovery_handle = object()
    owned_error = WorkloadStillOwnedError("ownership retained", recovery_handle)
    monkeypatch.setattr(
        svc,
        "run_training",
        lambda *args, **kwargs: (_ for _ in ()).throw(owned_error),
    )

    with pytest.raises(WorkloadStillOwnedError) as raised:
        svc.TrainingOrchestrator(tmp_path / "workspace").run_role_training(
            _registered_spec(tmp_path)
        )

    assert raised.value is owned_error
    assert raised.value.sidecar is recovery_handle
    assert raised.value.run_id
    record = load_registry()["runs"][0]
    assert record["status"] == "recovery-required"
    assert not record["finished_at"]
    assert record["failure_kind"] == "workload-still-owned"
    assert record["containment"]["ownership"] == "retained"


def test_registry_failure_never_replaces_owned_workload_error(monkeypatch, tmp_path):
    import hydra_suite.training.registry as registry
    from hydra_suite.runtime.process_supervisor import WorkloadStillOwnedError

    monkeypatch.setattr(registry, "_project_root", lambda: tmp_path)
    recovery_handle = object()
    owned_error = WorkloadStillOwnedError("ownership retained", recovery_handle)
    monkeypatch.setattr(
        svc,
        "run_training",
        lambda *args, **kwargs: (_ for _ in ()).throw(owned_error),
    )
    monkeypatch.setattr(
        svc,
        "update_run_record",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("registry offline")),
    )

    with pytest.raises(WorkloadStillOwnedError) as raised:
        svc.TrainingOrchestrator(tmp_path / "workspace").run_role_training(
            _registered_spec(tmp_path)
        )

    assert raised.value is owned_error
    assert raised.value.sidecar is recovery_handle
    assert raised.value.registry_update_error == "registry offline"


def test_auto_publish_exception_finalizes_registry_with_training_diagnostics(
    monkeypatch, tmp_path
):
    import hydra_suite.training.registry as registry

    monkeypatch.setattr(registry, "_project_root", lambda: tmp_path)
    artifact = tmp_path / "adapters.pt"
    artifact.write_bytes(b"adapter")
    monkeypatch.setattr(
        svc,
        "run_training",
        lambda *args, **kwargs: {
            "success": True,
            "artifact_path": str(artifact),
            "resource_preflight": "/tmp/resource_preflight.json",
            "containment": {"backend": "systemd", "peak_tree_rss_bytes": 123},
        },
    )
    monkeypatch.setattr(
        svc,
        "_publish_training_artifacts",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("corrupt adapter")),
    )

    result = svc.TrainingOrchestrator(tmp_path / "workspace").run_role_training(
        _registered_spec(tmp_path, auto_import=True)
    )

    assert result["success"] is False
    assert result["run_id"]
    assert result["failure_kind"] == "publish-exception"
    assert result["resource_preflight"] == "/tmp/resource_preflight.json"
    assert result["containment"]["backend"] == "systemd"
    record = load_registry()["runs"][0]
    assert record["status"] == "failed"
    assert record["failure_kind"] == "publish-exception"
    assert record["resource_preflight"] == "/tmp/resource_preflight.json"
    assert record["containment"]["peak_tree_rss_bytes"] == 123


def test_returned_resource_refusal_is_preserved_in_final_registry_record(
    monkeypatch, tmp_path
):
    import hydra_suite.training.registry as registry

    monkeypatch.setattr(registry, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        svc,
        "run_training",
        lambda *args, **kwargs: {
            "success": False,
            "error": "insufficient host memory",
            "failure_kind": "host-admission-refusal",
            "resource_preflight": "/tmp/preflight.json",
            "containment": {"backend": "rlimit_as", "host_limit_bytes": 456},
        },
    )

    result = svc.TrainingOrchestrator(tmp_path / "workspace").run_role_training(
        _registered_spec(tmp_path)
    )

    assert result["run_id"]
    record = load_registry()["runs"][0]
    assert record["status"] == "failed"
    assert record["error_message"] == "insufficient host memory"
    assert record["failure_kind"] == "host-admission-refusal"
    assert record["resource_preflight"] == "/tmp/preflight.json"
    assert record["containment"]["host_limit_bytes"] == 456
