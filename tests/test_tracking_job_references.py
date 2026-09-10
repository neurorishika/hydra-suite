"""Copying a model reference into a job, with sidecars, bundles and registry."""

import hashlib
import json

import pytest

from hydra_suite.data.tracking_job.references import (
    PlannedModel,
    copy_model_reference,
    external_key_for,
    write_registry_subset,
)


@pytest.fixture()
def source_root(tmp_path):
    root = tmp_path / "hostmodels"
    (root / "obb").mkdir(parents=True)
    model = root / "obb" / "x.pt"
    model.write_bytes(b"weights")
    (root / "obb" / "x.pt.slice_meta.json").write_text('{"slice": 1}')
    (root / "obb" / "x.pt.canonical_meta.json").write_text('{"canonical": 1}')
    return root


def test_file_model_is_copied_with_its_sidecars(source_root, tmp_path):
    models_root = tmp_path / "job" / "models"
    planned = PlannedModel(
        role="YOLO_OBB_DIRECT_MODEL_PATH",
        source_path=str(source_root / "obb" / "x.pt"),
        kind="file",
        key="obb/x.pt",
    )
    record = copy_model_reference(planned, models_root)
    assert (models_root / "obb" / "x.pt").read_bytes() == b"weights"
    assert (models_root / "obb" / "x.pt.slice_meta.json").exists()
    assert (models_root / "obb" / "x.pt.canonical_meta.json").exists()
    assert set(record.sidecars) == {
        "obb/x.pt.slice_meta.json",
        "obb/x.pt.canonical_meta.json",
    }
    assert record.kind == "file"
    assert record.sha256 and record.size_bytes == 7


def test_sha256_matches_the_file_bytes(source_root, tmp_path):
    # Fix W9: `hashlib` is used by several tests below (directory/bundle
    # digest assertions) without a local import -- it must be imported once
    # at module level (done in this file's header above), not re-imported
    # locally here, or every OTHER test using it raises NameError.
    planned = PlannedModel(
        role="R",
        source_path=str(source_root / "obb" / "x.pt"),
        kind="file",
        key="obb/x.pt",
    )
    record = copy_model_reference(planned, tmp_path / "models")
    assert record.sha256 == hashlib.sha256(b"weights").hexdigest()


def test_directory_model_is_copied_whole(tmp_path):
    src = tmp_path / "host" / "pose" / "SLEAP" / "run"
    src.mkdir(parents=True)
    (src / "best.ckpt").write_bytes(b"c")
    (src / "training_config.json").write_text("{}")
    nested = src / "sub"
    nested.mkdir()
    (nested / "extra.json").write_text("{}")
    planned = PlannedModel(
        role="POSE_MODEL_DIR",
        source_path=str(src),
        kind="directory",
        key="pose/SLEAP/run",
    )
    models_root = tmp_path / "job" / "models"
    record = copy_model_reference(planned, models_root)
    assert (models_root / "pose" / "SLEAP" / "run" / "best.ckpt").exists()
    assert (models_root / "pose" / "SLEAP" / "run" / "sub" / "extra.json").exists()
    assert record.kind == "directory"
    assert "pose/SLEAP/run/best.ckpt" in record.files
    # Fix B4: file_digests must be POPULATED, not merely declared. Without
    # these assertions the field ships empty and verify_job's member-integrity
    # branch (which is gated on `if model.file_digests:`) never runs at all.
    assert set(record.file_digests) == set(record.files)
    assert (
        record.file_digests["pose/SLEAP/run/best.ckpt"]
        == hashlib.sha256(b"c").hexdigest()
    )
    # A directory model carries no top-level digest/size; verify_job must skip
    # its sha256/size_bytes checks for kind == "directory".
    assert record.sha256 == ""
    assert record.size_bytes == 0


def test_directory_model_excludes_host_runtime_artifacts(tmp_path):
    src = tmp_path / "host" / "run"
    (src / ".hydra-runtime-artifacts").mkdir(parents=True)
    (src / "best.ckpt").write_bytes(b"c")
    (src / ".hydra-runtime-artifacts" / "m.engine").write_bytes(b"host")
    models_root = tmp_path / "job" / "models"
    record = copy_model_reference(
        PlannedModel(
            role="POSE_MODEL_DIR",
            source_path=str(src),
            kind="directory",
            key="pose/run",
        ),
        models_root,
    )
    assert not (models_root / "pose" / "run" / ".hydra-runtime-artifacts").exists()
    assert not any(".hydra-runtime-artifacts" in f for f in record.files)
    assert not any(".hydra-runtime-artifacts" in f for f in record.file_digests)


def test_bundle_artifacts_are_copied_beside_the_selected_checkpoint(tmp_path):
    src = tmp_path / "host" / "classification" / "identity"
    src.mkdir(parents=True)
    head_a = src / "head_a.pth"
    head_b = src / "head_b.pth"
    manifest = src / "ids.bundle.json"
    head_a.write_bytes(b"a")
    head_b.write_bytes(b"b")
    manifest.write_text("{}")
    planned = PlannedModel(
        role="CNN_CLASSIFIERS",
        source_path=str(head_a),
        kind="file",
        key="classification/identity/head_a.pth",
        bundle_artifacts=[str(head_b), str(manifest)],
    )
    models_root = tmp_path / "job" / "models"
    record = copy_model_reference(planned, models_root)
    assert (models_root / "classification" / "identity" / "head_b.pth").exists()
    assert (models_root / "classification" / "identity" / "ids.bundle.json").exists()
    assert "classification/identity/head_b.pth" in record.files
    # Fix B4: bundle artifacts are separate files the primary's sha256 cannot
    # cover, so each must carry its own digest.
    assert set(record.file_digests) == set(record.files)
    assert (
        record.file_digests["classification/identity/head_b.pth"]
        == hashlib.sha256(b"b").hexdigest()
    )
    assert (
        record.file_digests["classification/identity/ids.bundle.json"]
        == hashlib.sha256(b"{}").hexdigest()
    )


def test_bundle_head_sidecar_is_recorded_so_it_gets_pushed(tmp_path):
    """Fix V1: a bundle head's own .v2meta.json must land in record.sidecars,
    not just get copied to disk -- Task 9's build_push_input_list only walks
    manifest.models[*].sidecars/files, so an un-recorded sidecar never ships,
    and a missing .v2meta.json for a flat .pt classifier silently falls back
    to fit_policy 'squash' on the remote (backend.py:38-52, :288)."""
    src = tmp_path / "host" / "classification" / "identity"
    src.mkdir(parents=True)
    head_a = src / "head_a.pth"
    head_b = src / "head_b.pth"
    head_a.write_bytes(b"a")
    head_b.write_bytes(b"b")
    # Fix X9 (round-6): the real convention is `.with_suffix(".v2meta.json")`
    # (REPLACES the model's own suffix), not an appended
    # "<name>.pth.v2meta.json" -- verified core/inference/model_paths.py:44
    # (`src.with_suffix(".v2meta.json")`) and
    # core/individual/classification/backend.py:288, which is the actual
    # reader. The original test wrote and asserted a name the real code never
    # produces or looks for, so it proved nothing about the file
    # copy_model_metadata_sidecars/backend.py actually round-trip.
    (src / "head_b.v2meta.json").write_text('{"fit_policy": "letterbox"}')
    planned = PlannedModel(
        role="CNN_CLASSIFIERS",
        source_path=str(head_a),
        kind="file",
        key="classification/identity/head_a.pth",
        bundle_artifacts=[str(head_b)],
    )
    models_root = tmp_path / "job" / "models"
    record = copy_model_reference(planned, models_root)
    assert "classification/identity/head_b.v2meta.json" in record.sidecars


def test_missing_source_fails_with_the_role_and_the_path(tmp_path):
    from hydra_suite.data.tracking_job.manifest import TrackingJobError

    planned = PlannedModel(
        role="YOLO_HEADTAIL_MODEL_PATH",
        source_path=str(tmp_path / "gone.pt"),
        kind="file",
        key="x/gone.pt",
    )
    with pytest.raises(TrackingJobError) as excinfo:
        copy_model_reference(planned, tmp_path / "models")
    message = str(excinfo.value)
    assert "YOLO_HEADTAIL_MODEL_PATH" in message and "gone.pt" in message


def test_external_key_is_deterministic_and_namespaced(tmp_path):
    outside = tmp_path / "elsewhere" / "m.pt"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"z")
    key = external_key_for(str(outside))
    assert key.startswith("external/") and key.endswith("/m.pt")
    assert external_key_for(str(outside)) == key


def test_copy_is_idempotent(source_root, tmp_path):
    planned = PlannedModel(
        role="R",
        source_path=str(source_root / "obb" / "x.pt"),
        kind="file",
        key="obb/x.pt",
    )
    models_root = tmp_path / "models"
    first = copy_model_reference(planned, models_root)
    second = copy_model_reference(planned, models_root)
    assert first == second


def test_registry_subset_contains_only_shipped_keys_with_null_source(tmp_path):
    entries = [
        ("obb/x.pt", {"species": "ant", "source_path": "/host/train/run/best.pt"}),
        ("obb/unused.pt", {"species": "fly", "source_path": "/host/other.pt"}),
    ]
    destination = tmp_path / "models" / "model_registry.json"
    count = write_registry_subset({"obb/x.pt"}, entries, destination)
    payload = json.loads(destination.read_text())
    assert count == 1
    assert payload["schema_version"] == 2
    assert set(payload["entries"]) == {"obb/x.pt"}
    assert payload["entries"]["obb/x.pt"]["source_path"] is None
    assert payload["entries"]["obb/x.pt"]["species"] == "ant"


def test_registry_subset_writes_an_empty_v2_registry_when_nothing_matches(tmp_path):
    destination = tmp_path / "models" / "model_registry.json"
    assert write_registry_subset(set(), [], destination) == 0
    payload = json.loads(destination.read_text())
    assert payload == {"schema_version": 2, "entries": {}}
