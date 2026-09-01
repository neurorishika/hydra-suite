"""The role's params must reach the builder; nothing else may change."""

import inspect

from hydra_suite.training.dataset_builders import prepare_role_dataset
from hydra_suite.training.service import TrainingOrchestrator


def test_prepare_role_dataset_accepts_sam3_params_seed_and_split():
    sig = inspect.signature(prepare_role_dataset)
    for name in ("sam3_params", "seed", "split"):
        assert name in sig.parameters, f"{name} missing"
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_orchestrator_forwards_them(monkeypatch, tmp_path):
    seen = {}

    def fake_prepare(role, merged_obb_dataset_dir, role_output_root, *a, **kw):
        seen.update(kw)
        from hydra_suite.training.contracts import DatasetBuildResult

        return DatasetBuildResult(dataset_dir=str(role_output_root))

    import hydra_suite.training.service as svc

    monkeypatch.setattr(svc, "prepare_role_dataset", fake_prepare)
    monkeypatch.setattr(
        svc,
        "validate_role_dataset",
        lambda *a, **k: __import__(
            "hydra_suite.training.contracts", fromlist=["x"]
        ).ValidationReport(valid=True),
    )

    from hydra_suite.training.contracts import Sam3LoraParams, TrainingRole

    orch = TrainingOrchestrator(tmp_path)
    params = Sam3LoraParams(prompt="ant")
    orch.build_role_dataset(
        TrainingRole.SEMANTIC_SAM3, str(tmp_path), sam3_params=params, seed=7
    )
    assert seen["sam3_params"] is params
    assert seen["seed"] == 7


def test_existing_roles_are_unaffected():
    sig = inspect.signature(prepare_role_dataset)
    # The pre-existing positional contract must not move.
    names = list(sig.parameters)
    assert names[:4] == [
        "role",
        "merged_obb_dataset_dir",
        "role_output_root",
        "class_name",
    ]
