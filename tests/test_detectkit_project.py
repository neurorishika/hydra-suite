"""Tests for DetectKit project model and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hydra_suite.data.project_bundle import (
    export_project_bundle_archive,
    import_project_bundle_archive,
)
from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource, StagedReview
from hydra_suite.detectkit.gui.project import (
    create_project,
    default_project_parent_dir,
    detectkit_artifact_paths,
    detectkit_model_path_is_previewable,
    detectkit_models_dir,
    detectkit_project_is_portable,
    detectkit_project_linked_reference_counts,
    detectkit_project_model_paths,
    detectkit_project_preview_model_paths,
    detectkit_resolve_inference_models,
    legacy_project_file_path,
    make_detectkit_project_portable,
    open_project,
    project_exists,
    project_file_path,
    record_training_results,
    save_project,
)


@pytest.fixture(autouse=True)
def _isolate_recent_projects_file(monkeypatch, tmp_path: Path) -> None:
    """Keep project-lifecycle tests from writing the user's GUI recents list."""
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.project.get_recent_projects_path",
        lambda: tmp_path / "recent_projects.json",
    )


def test_project_roundtrip(tmp_path: Path):
    proj = DetectKitProject(
        project_dir=tmp_path,
        class_names=["ant", "bee"],
        sources=[
            OBBSource(path=str(tmp_path / "ds1"), name="ds1"),
            OBBSource(path=str(tmp_path / "ds2"), name="ds2"),
        ],
    )
    proj_file = tmp_path / "detectkit_project.json"
    proj.save(proj_file)
    assert proj_file.exists()

    loaded = DetectKitProject.load(proj_file)
    assert loaded.class_name == "ant"
    assert loaded.class_names == ["ant", "bee"]
    assert len(loaded.sources) == 2
    assert loaded.sources[0].name == "ds1"


def test_project_save_is_atomic_when_replacement_fails(tmp_path: Path, monkeypatch):
    project_file = tmp_path / "detectkit_project.json"
    project = DetectKitProject(project_dir=tmp_path, class_names=["before"])
    project.save(project_file)
    original = project_file.read_bytes()
    project.class_names = ["after"]

    original_replace = Path.replace

    def fail_temp_replace(path: Path, target: Path):
        if path.suffix == ".tmp" and Path(target) == project_file:
            raise OSError("simulated replace failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_temp_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        project.save(project_file)

    assert project_file.read_bytes() == original


def test_project_roundtrip_preserves_semantic_settings_and_calibration(tmp_path: Path):
    proj = DetectKitProject(project_dir=tmp_path)
    proj.semantic_escalation_settings = {
        "prompt": "ant",
        "confidence": 0.42,
        "tile_fraction": 0.08,
    }
    proj.semantic_calibration = {
        "created_at": "2026-08-30T12:00:00+00:00",
        "points": [{"confidence": 0.42, "tile_fraction": 0.08}],
    }

    project_file = tmp_path / "detectkit_project.json"
    proj.save(project_file)
    loaded = DetectKitProject.load(project_file)

    assert loaded.semantic_escalation_settings == proj.semantic_escalation_settings
    assert loaded.semantic_calibration == proj.semantic_calibration


def test_project_loads_legacy_single_class_field(tmp_path: Path):
    proj_file = tmp_path / "detectkit_project.json"
    proj_file.write_text(
        json.dumps(
            {
                "version": 1,
                "project_dir": str(tmp_path),
                "class_name": "ant",
                "sources": [],
            }
        ),
        encoding="utf-8",
    )

    loaded = DetectKitProject.load(proj_file)

    assert loaded.class_name == "ant"
    assert loaded.class_names == ["ant"]


def test_project_defaults():
    proj = DetectKitProject(project_dir=Path("/tmp/test"))
    assert proj.class_name == "object"
    assert proj.class_names == ["object"]
    assert proj.sources == []
    assert proj.split_train == 0.8
    assert proj.split_val == 0.2
    assert proj.seed == 42


def test_obb_source_roundtrip():
    src = OBBSource(
        path="/data/obb_ds",
        name="my_dataset",
        original_path="/mnt/original/obb_ds",
        source_kind="yolo_detect",
        imported=True,
    )
    d = src.to_dict()
    restored = OBBSource.from_dict(d)
    assert restored.path == "/data/obb_ds"
    assert restored.name == "my_dataset"
    assert restored.original_path == "/mnt/original/obb_ds"
    assert restored.source_kind == "yolo_detect"
    assert restored.imported is True


def test_save_open_project_preserves_all_obb_source_fields(tmp_path: Path) -> None:
    """Regression: save_project/open_project must not drop OBBSource fields.

    `_serialize_project_state_paths`/`_deserialize_project_state_paths` used
    to rebuild each source from six hand-listed fields, silently discarding
    level/reviewed/derived_from/sam2_variant/staged_review -- so a
    project saved right after a SAM2 escalation staged a result lost the
    staging record on the very next save (orphaning the staging directory),
    and an accepted escalation's level="polygon"/reviewed=False reverted to
    obb/reviewed on reload.
    """
    proj = create_project(tmp_path, class_names=["ant", "bee"])
    source_dir = tmp_path / "artifacts" / "imported_sources" / "ds1"
    source_dir.mkdir(parents=True, exist_ok=True)
    staged_dir = tmp_path / "artifacts" / "pending_escalations" / "ds1-sam2-abc123"
    staged_dir.mkdir(parents=True, exist_ok=True)
    pending = StagedReview(
        staged_path=str(staged_dir),
        target_level="polygon",
        producer="sam2",
        producer_variant="sam2.1-hiera-base_plus",
        created_at="2026-08-27T00:00:00",
    )
    proj.sources = [
        OBBSource(
            path=str(source_dir),
            name="ds1",
            validated=True,
            original_path="/elsewhere/ds1",
            source_kind="detectkit_al",
            imported=True,
            level="polygon",
            reviewed=False,
            derived_from="ds0",
            sam2_variant="sam2.1-hiera-base_plus",
            staged_review=pending,
        )
    ]
    save_project(proj)

    loaded = open_project(tmp_path)
    assert loaded is not None
    assert len(loaded.sources) == 1
    restored = loaded.sources[0]

    assert Path(restored.path) == source_dir
    assert restored.name == "ds1"
    assert restored.validated is True
    assert restored.original_path == "/elsewhere/ds1"
    assert restored.source_kind == "detectkit_al"
    assert restored.imported is True
    assert restored.level == "polygon"
    assert restored.reviewed is False
    assert restored.derived_from == "ds0"
    assert restored.sam2_variant == "sam2.1-hiera-base_plus"
    assert restored.staged_review is not None
    assert Path(restored.staged_review.staged_path) == staged_dir
    assert restored.staged_review.target_level == "polygon"
    assert restored.staged_review.producer_variant == "sam2.1-hiera-base_plus"
    assert restored.staged_review.created_at == "2026-08-27T00:00:00"


def test_default_project_parent_dir_uses_hydra_projects_root(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("HYDRA_PROJECTS_DIR", str(tmp_path / "hydra-projects"))

    assert default_project_parent_dir() == tmp_path / "hydra-projects" / "DetectKit"


def test_create_project_uses_bundle_layout(tmp_path: Path) -> None:
    proj = create_project(tmp_path, class_names=["ant", "bee"])
    artifact_paths = detectkit_artifact_paths(tmp_path)

    assert proj.project_dir == tmp_path.resolve()
    assert (tmp_path / "hydra_project.json").exists()
    assert (tmp_path / "state").is_dir()
    assert (tmp_path / "artifacts").is_dir()
    assert (tmp_path / "history").is_dir()
    assert detectkit_models_dir(tmp_path).is_dir()
    assert artifact_paths["training_runs"].is_dir()
    assert artifact_paths["evaluation"].is_dir()
    assert artifact_paths["exports"].is_dir()
    assert project_file_path(tmp_path).exists()
    assert not legacy_project_file_path(tmp_path).exists()
    assert project_exists(tmp_path) is True


def test_open_project_migrates_legacy_root_file_to_bundle(tmp_path: Path) -> None:
    legacy_path = legacy_project_file_path(tmp_path)
    legacy_path.write_text(
        json.dumps(
            {
                "version": 1,
                "project_dir": str(tmp_path),
                "class_names": ["ant", "bee"],
                "sources": [{"path": "/data/ds1", "name": "ds1"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    loaded = open_project(tmp_path)

    assert loaded is not None
    assert loaded.class_names == ["ant", "bee"]
    assert loaded.sources[0].name == "ds1"
    assert (tmp_path / "hydra_project.json").exists()
    assert project_file_path(tmp_path).exists()
    assert not legacy_path.exists()
    assert (tmp_path / "history" / "legacy_detectkit_project.json").exists()


def test_open_project_reads_bundle_manifest(tmp_path: Path) -> None:
    created = create_project(tmp_path, class_names=["ant", "bee"])

    loaded = open_project(tmp_path)

    assert loaded is not None
    assert loaded.project_dir == created.project_dir
    assert loaded.class_names == ["ant", "bee"]


def test_open_project_recovers_from_malformed_manifest_using_legacy_file(
    tmp_path: Path,
) -> None:
    (tmp_path / "hydra_project.json").write_text("{bad-manifest", encoding="utf-8")
    legacy_path = legacy_project_file_path(tmp_path)
    legacy_path.write_text(
        json.dumps(
            {
                "version": 1,
                "project_dir": str(tmp_path),
                "class_names": ["ant"],
                "sources": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    loaded = open_project(tmp_path)

    assert loaded is not None
    assert loaded.class_names == ["ant"]
    assert project_file_path(tmp_path).exists()
    assert (tmp_path / "history" / "legacy_detectkit_project.json").exists()


def test_record_training_results_exports_models_and_logs_to_project(
    tmp_path: Path,
) -> None:
    project = create_project(tmp_path / "project", class_names=["ant"])
    run_dir = tmp_path / "workspace_run"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    artifact_path = weights_dir / "best.pt"
    artifact_path.write_bytes(b"weights")
    published_path = tmp_path / "published.pt"
    published_path.write_bytes(b"weights")
    published_path.with_suffix(".pt.slice_meta.json").write_text(
        json.dumps({"geometry_mode": "auto_object", "reference_body_px": 42.0}),
        encoding="utf-8",
    )
    metrics_path = run_dir / "results.csv"
    metrics_path.write_text("epoch,metric\n1,0.9\n", encoding="utf-8")

    persisted = record_training_results(
        project,
        [
            {
                "run_id": "20260420-120000_obb_direct_deadbeef",
                "role": "obb_direct",
                "success": True,
                "artifact_path": str(artifact_path),
                "published_model_path": str(published_path),
                "metrics_path": str(metrics_path),
                "training_log": "epoch 1\nmetric 0.9",
            }
        ],
    )

    assert len(persisted) == 1
    entry = persisted[0]
    assert Path(entry["project_model_path"]).exists()
    assert Path(entry["project_model_path"]).parent == project.project_dir / "models"
    exported_sidecar = Path(entry["project_model_path"]).with_suffix(
        ".pt.slice_meta.json"
    )
    assert json.loads(exported_sidecar.read_text(encoding="utf-8")) == {
        "geometry_mode": "auto_object",
        "reference_body_px": 42.0,
    }
    assert Path(entry["project_log_path"]).exists()
    assert Path(entry["project_metrics_paths"][0]).exists()
    assert detectkit_project_model_paths(project) == [entry["project_model_path"]]


def test_detectkit_project_model_paths_prefers_active_model(tmp_path: Path) -> None:
    project = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    model_a = tmp_path / "models" / "a.pt"
    model_b = tmp_path / "models" / "b.pt"
    model_a.parent.mkdir(parents=True)
    model_a.write_bytes(b"a")
    model_b.write_bytes(b"b")
    project.active_model_path = str(model_b)
    project.training_history = [
        {"run_id": "run_1", "project_model_path": str(model_a)},
        {"run_id": "run_2", "project_model_path": str(model_b)},
    ]

    assert detectkit_project_model_paths(project) == [str(model_b), str(model_a)]


def test_detectkit_project_preview_model_paths_filters_non_preview_roles(
    tmp_path: Path,
) -> None:
    project = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    model_obb = tmp_path / "models" / "obb.pt"
    model_seq = tmp_path / "models" / "seq.pt"
    model_obb.parent.mkdir(parents=True)
    model_obb.write_bytes(b"obb")
    model_seq.write_bytes(b"seq")
    project.training_history = [
        {
            "run_id": "run_1",
            "role": "obb_direct",
            "project_model_path": str(model_obb),
        },
        {
            "run_id": "run_2",
            "role": "seq_detect",
            "project_model_path": str(model_seq),
        },
    ]

    assert detectkit_project_preview_model_paths(project) == [str(model_obb)]
    assert detectkit_model_path_is_previewable(project, str(model_obb)) is True
    assert detectkit_model_path_is_previewable(project, str(model_seq)) is False


def test_record_training_results_prefers_previewable_active_model(
    tmp_path: Path,
) -> None:
    project = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    project.auto_select = True

    obb_artifact = tmp_path / "runs" / "obb" / "weights" / "best.pt"
    seq_artifact = tmp_path / "runs" / "seq" / "weights" / "best.pt"
    obb_artifact.parent.mkdir(parents=True)
    seq_artifact.parent.mkdir(parents=True)
    obb_artifact.write_bytes(b"obb")
    seq_artifact.write_bytes(b"seq")

    record_training_results(
        project,
        [
            {
                "run_id": "run_obb",
                "role": "obb_direct",
                "success": True,
                "artifact_path": str(obb_artifact),
            },
            {
                "run_id": "run_seq",
                "role": "seq_detect",
                "success": True,
                "artifact_path": str(seq_artifact),
            },
        ],
    )

    assert project.active_model_path
    assert detectkit_model_path_is_previewable(project, project.active_model_path)


def test_detectkit_project_portability_helpers_count_external_references(
    tmp_path: Path,
) -> None:
    project = create_project(tmp_path / "project", class_names=["ant"])
    local_source = project.project_dir / "artifacts" / "imported_sources" / "source-a"
    local_source.mkdir(parents=True, exist_ok=True)
    local_model = project.project_dir / "models" / "best.pt"
    local_model.parent.mkdir(parents=True, exist_ok=True)
    local_model.write_bytes(b"weights")
    external_source = tmp_path / "external" / "source-b"
    external_source.mkdir(parents=True, exist_ok=True)
    external_model = tmp_path / "external" / "best.pt"
    external_model.write_bytes(b"weights")

    project.sources = [
        OBBSource(path=str(local_source), name="local"),
        OBBSource(path=str(external_source), name="external"),
    ]
    project.active_model_path = str(local_model)
    project.training_history = [
        {
            "run_id": "run_1",
            "project_model_path": str(external_model),
        }
    ]

    assert detectkit_project_linked_reference_counts(project) == {
        "sources": 1,
        "artifacts": 1,
    }
    assert detectkit_project_is_portable(project) is False

    project.sources = [OBBSource(path=str(local_source), name="local")]
    project.training_history = [
        {"run_id": "run_1", "project_model_path": str(local_model)}
    ]

    assert detectkit_project_linked_reference_counts(project) == {
        "sources": 0,
        "artifacts": 0,
    }
    assert detectkit_project_is_portable(project) is True


def test_make_detectkit_project_portable_localizes_sources_and_artifacts(
    tmp_path: Path,
) -> None:
    project = create_project(tmp_path / "project", class_names=["ant"])
    external_source = tmp_path / "external_source"
    (external_source / "images").mkdir(parents=True, exist_ok=True)
    (external_source / "labels").mkdir(parents=True, exist_ok=True)
    (external_source / "images" / "frame_001.png").write_bytes(b"png")
    (external_source / "labels" / "frame_001.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n",
        encoding="utf-8",
    )
    (external_source / "classes.txt").write_text("ant\n", encoding="utf-8")

    external_model = tmp_path / "external_model.pt"
    external_model.write_bytes(b"weights")
    external_metrics = tmp_path / "results.csv"
    external_metrics.write_text("epoch,metric\n1,0.9\n", encoding="utf-8")
    external_log = tmp_path / "training.log"
    external_log.write_text("epoch 1\n", encoding="utf-8")

    project.sources = [OBBSource(path=str(external_source), name="external")]
    project.active_model_path = str(external_model)
    project.training_history = [
        {
            "run_id": "run_1",
            "artifact_paths": [str(external_model)],
            "project_model_path": str(external_model),
            "project_model_paths": [str(external_model)],
            "project_metrics_paths": [str(external_metrics)],
            "project_log_path": str(external_log),
            "project_run_dir": str(tmp_path / "external_run"),
        }
    ]

    linked_counts = make_detectkit_project_portable(project)

    assert linked_counts == {"sources": 0, "artifacts": 0}
    assert detectkit_project_is_portable(project) is True
    assert Path(project.sources[0].path).is_relative_to(project.project_dir.resolve())
    assert project.sources[0].imported is True
    assert Path(project.active_model_path).is_relative_to(project.project_dir.resolve())
    entry = project.training_history[0]
    assert Path(entry["project_model_path"]).is_relative_to(
        project.project_dir.resolve()
    )
    assert Path(entry["project_metrics_paths"][0]).is_relative_to(
        project.project_dir.resolve()
    )
    assert Path(entry["project_log_path"]).is_relative_to(project.project_dir.resolve())


def test_detectkit_project_owned_paths_round_trip_through_archive(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "detectkit_project"
    project = create_project(project_dir, class_names=["ant"])
    source_dir = project_dir / "artifacts" / "imported_sources" / "source-a"
    source_dir.mkdir(parents=True, exist_ok=True)

    model_path = project_dir / "models" / "latest.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"weights")

    run_dir = project_dir / "artifacts" / "training_runs" / "run-a"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "results.csv"
    metrics_path.write_text("epoch,metric\n1,0.9\n", encoding="utf-8")
    log_path = run_dir / "training.log"
    log_path.write_text("epoch 1\nmetric 0.9\n", encoding="utf-8")

    project.sources = [
        OBBSource(
            path=str(source_dir.resolve()),
            name="source-a",
            original_path="/external/source-a",
            source_kind="detectkit",
            imported=True,
        )
    ]
    project.active_model_path = str(model_path.resolve())
    project.training_history = [
        {
            "run_id": "run-a",
            "project_model_path": str(model_path.resolve()),
            "project_model_paths": [str(model_path.resolve())],
            "project_run_dir": str(run_dir.resolve()),
            "project_metrics_paths": [str(metrics_path.resolve())],
            "project_log_path": str(log_path.resolve()),
        }
    ]

    save_project(project)

    raw = json.loads(project_file_path(project_dir).read_text(encoding="utf-8"))
    assert raw["sources"][0]["path"] == "artifacts/imported_sources/source-a"
    assert raw["active_model_path"] == "models/latest.pt"
    assert raw["training_history"][0]["project_model_path"] == "models/latest.pt"
    assert (
        raw["training_history"][0]["project_run_dir"] == "artifacts/training_runs/run-a"
    )
    assert raw["training_history"][0]["project_metrics_paths"] == [
        "artifacts/training_runs/run-a/results.csv"
    ]

    archive_path = export_project_bundle_archive(
        project_dir, tmp_path / "detectkit.zip"
    )
    restored_dir = import_project_bundle_archive(
        archive_path,
        tmp_path / "restored_detectkit_project",
        expected_kit="detectkit",
    )

    loaded = open_project(restored_dir)

    assert loaded is not None
    assert loaded.sources[0].path == str(
        (restored_dir / "artifacts" / "imported_sources" / "source-a").resolve()
    )
    assert loaded.active_model_path == str(
        (restored_dir / "models" / "latest.pt").resolve()
    )
    assert loaded.training_history[0]["project_model_path"] == str(
        (restored_dir / "models" / "latest.pt").resolve()
    )
    assert loaded.training_history[0]["project_run_dir"] == str(
        (restored_dir / "artifacts" / "training_runs" / "run-a").resolve()
    )


# ---------------------------------------------------------------------------
# detectkit_resolve_inference_models tests
# ---------------------------------------------------------------------------


def _make_model_file(tmp_path: Path, name: str) -> str:
    p = tmp_path / name
    p.write_bytes(b"weights")
    return str(p)


def test_resolve_inference_models_obb_direct(tmp_path: Path) -> None:
    model = _make_model_file(tmp_path, "best_obb.pt")
    project = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    project.training_history = [
        {
            "run_id": "run_1",
            "role": "obb_direct",
            "project_model_path": model,
        }
    ]

    kind, primary, secondary = detectkit_resolve_inference_models(project, model)

    assert kind == "obb_direct"
    assert primary == model
    assert secondary is None


@pytest.mark.parametrize("role", ["detect_direct", "segment_direct"])
def test_resolve_inference_models_direct_non_obb_roles(
    tmp_path: Path, role: str
) -> None:
    """Every direct detector head can be selected for inference."""
    model = _make_model_file(tmp_path, f"best_{role}.pt")
    project = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    project.training_history = [
        {"run_id": "run_1", "role": role, "project_model_path": model}
    ]

    kind, primary, secondary = detectkit_resolve_inference_models(project, model)

    assert kind == role
    assert primary == model
    assert secondary is None
    assert detectkit_model_path_is_previewable(project, model)


def test_resolve_inference_models_sequential_via_detect(tmp_path: Path) -> None:
    detect_model = _make_model_file(tmp_path, "detect.pt")
    obb_model = _make_model_file(tmp_path, "obb_crop.pt")
    project = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    project.training_history = [
        {
            "run_id": "run_detect",
            "role": "seq_detect",
            "project_model_path": detect_model,
        },
        {
            "run_id": "run_obb",
            "role": "seq_crop_obb",
            "project_model_path": obb_model,
        },
    ]

    kind, primary, secondary = detectkit_resolve_inference_models(project, detect_model)

    assert kind == "sequential"
    assert primary == detect_model
    assert secondary == obb_model


def test_resolve_inference_models_sequential_via_obb(tmp_path: Path) -> None:
    detect_model = _make_model_file(tmp_path, "detect.pt")
    obb_model = _make_model_file(tmp_path, "obb_crop.pt")
    project = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    project.training_history = [
        {
            "run_id": "run_detect",
            "role": "seq_detect",
            "project_model_path": detect_model,
        },
        {
            "run_id": "run_obb",
            "role": "seq_crop_obb",
            "project_model_path": obb_model,
        },
    ]

    # Choosing the obb path as primary — should normalize to (detect, obb) ordering.
    kind, primary, secondary = detectkit_resolve_inference_models(project, obb_model)

    assert kind == "sequential"
    assert primary == detect_model
    assert secondary == obb_model


def test_resolve_inference_models_sequential_segment(tmp_path: Path) -> None:
    detect_model = _make_model_file(tmp_path, "detect.pt")
    segment_model = _make_model_file(tmp_path, "segment_crop.pt")
    project = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    project.training_history = [
        {
            "run_id": "run_detect",
            "role": "seq_detect",
            "project_model_path": detect_model,
        },
        {
            "run_id": "run_segment",
            "role": "seq_crop_segment",
            "project_model_path": segment_model,
        },
    ]

    kind, primary, secondary = detectkit_resolve_inference_models(
        project, segment_model
    )

    assert kind == "sequential_segment"
    assert primary == detect_model
    assert secondary == segment_model
    assert detectkit_model_path_is_previewable(project, segment_model)


def test_resolve_inference_models_missing_counterpart_raises(tmp_path: Path) -> None:
    detect_model = _make_model_file(tmp_path, "detect.pt")
    project = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    project.training_history = [
        {
            "run_id": "run_detect",
            "role": "seq_detect",
            "project_model_path": detect_model,
        }
        # No seq_crop_obb entry!
    ]

    import pytest

    with pytest.raises(RuntimeError, match="seq_crop_obb"):
        detectkit_resolve_inference_models(project, detect_model)


def test_resolve_inference_models_unknown_path(tmp_path: Path) -> None:
    # Path not in any training history entry.
    model = _make_model_file(tmp_path, "external.pt")
    project = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    project.training_history = []

    kind, primary, secondary = detectkit_resolve_inference_models(project, model)

    # Empty history role maps to obb_direct.
    assert kind == "obb_direct"
    assert primary == model
    assert secondary is None
