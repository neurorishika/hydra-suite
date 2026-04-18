# tests/test_classkit_publish.py
from unittest.mock import patch

from hydra_suite.training.contracts import TrainingRole
from hydra_suite.training.model_publish import _repo_dir_for_role


def test_new_roles_exist():
    assert TrainingRole.CLASSIFY_FLAT_YOLO.value == "classify_flat_yolo"
    assert TrainingRole.CLASSIFY_FLAT_TINY.value == "classify_flat_tiny"
    assert TrainingRole.CLASSIFY_MULTIHEAD_YOLO.value == "classify_multihead_yolo"
    assert TrainingRole.CLASSIFY_MULTIHEAD_TINY.value == "classify_multihead_tiny"


def test_repo_dir_flat_yolo(tmp_path):
    with patch(
        "hydra_suite.training.model_publish.get_models_root", return_value=tmp_path
    ):
        d = _repo_dir_for_role(TrainingRole.CLASSIFY_FLAT_YOLO, scheme_name="color2")
    assert d == tmp_path / "YOLO-classify" / "color2"
    assert d.exists()


def test_repo_dir_flat_tiny(tmp_path):
    with patch(
        "hydra_suite.training.model_publish.get_models_root", return_value=tmp_path
    ):
        d = _repo_dir_for_role(TrainingRole.CLASSIFY_FLAT_TINY, scheme_name="age")
    assert d == tmp_path / "tiny-classify" / "age"


def test_repo_dir_multihead_yolo(tmp_path):
    with patch(
        "hydra_suite.training.model_publish.get_models_root", return_value=tmp_path
    ):
        d = _repo_dir_for_role(
            TrainingRole.CLASSIFY_MULTIHEAD_YOLO, scheme_name="color2"
        )
    assert d == tmp_path / "YOLO-classify" / "multihead" / "color2"


def test_repo_dir_multihead_tiny(tmp_path):
    with patch(
        "hydra_suite.training.model_publish.get_models_root", return_value=tmp_path
    ):
        d = _repo_dir_for_role(
            TrainingRole.CLASSIFY_MULTIHEAD_TINY, scheme_name="color2"
        )
    assert d == tmp_path / "tiny-classify" / "multihead" / "color2"


def test_publish_trained_model_includes_scheme_metadata(tmp_path):
    """publish_trained_model accepts and stores scheme_name, factor_index, factor_name."""
    import json
    from unittest.mock import patch

    from hydra_suite.training.model_publish import publish_trained_model

    # Create a fake artifact
    artifact = tmp_path / "best.pth"
    artifact.write_bytes(b"fake")

    registry_path = tmp_path / "model_registry.json"

    with (
        patch(
            "hydra_suite.training.model_publish.get_models_root",
            return_value=tmp_path,
        ),
        patch(
            "hydra_suite.training.model_publish._registry_path",
            return_value=registry_path,
        ),
    ):
        key, stored = publish_trained_model(
            role=TrainingRole.CLASSIFY_FLAT_TINY,
            artifact_path=str(artifact),
            size="tiny",
            species="drosophila",
            model_info="color2_flat",
            trained_from_run_id="run_001",
            dataset_fingerprint="abc123",
            base_model="",
            scheme_name="color_tags_2factor",
            factor_index=None,
            factor_name=None,
        )

    registry = json.loads(registry_path.read_text())
    assert registry["schema_version"] == 2
    entry = registry["entries"][key]
    assert entry["scheme_name"] == "color_tags_2factor"
    assert entry["factor_index"] is None
    assert entry["factor_name"] is None


def test_task_usage_for_classify_roles():
    """CLASSIFY_* roles must resolve to explicit classify usage_role values."""
    from hydra_suite.training.model_publish import _task_usage_for_role

    assert _task_usage_for_role(TrainingRole.CLASSIFY_FLAT_YOLO) == (
        "classify",
        "classify_yolo",
    )
    assert _task_usage_for_role(TrainingRole.CLASSIFY_FLAT_TINY) == (
        "classify",
        "classify_tiny",
    )
    assert _task_usage_for_role(TrainingRole.CLASSIFY_MULTIHEAD_YOLO) == (
        "classify",
        "classify_yolo",
    )
    assert _task_usage_for_role(TrainingRole.CLASSIFY_MULTIHEAD_TINY) == (
        "classify",
        "classify_tiny",
    )
