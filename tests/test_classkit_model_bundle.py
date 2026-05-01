"""Tests for the ClassKit portable model bundle discovery."""

from __future__ import annotations

import json
from pathlib import Path

from hydra_suite.classkit.model_bundle import (
    MODEL_BUNDLE_MANIFEST_SUFFIX,
    discover_multihead_model_bundle,
    write_model_bundle_manifest,
)


def _write_dummy_artifacts(directory: Path, names: list[str]) -> list[Path]:
    paths: list[Path] = []
    for name in names:
        path = directory / name
        path.write_bytes(b"weights")
        paths.append(path)
    return paths


def test_discover_bundle_from_manifest_path_directly(tmp_path: Path) -> None:
    """Selecting a *.bundle.json should resolve the multi-head bundle directly."""
    bundle_dir = tmp_path / "multihead_yolo_run"
    bundle_dir.mkdir()
    artifacts = _write_dummy_artifacts(
        bundle_dir, ["factor_0_yolo.pt", "factor_1_yolo.pt"]
    )

    manifest_path = bundle_dir / f"factor_0_yolo{MODEL_BUNDLE_MANIFEST_SUFFIX}"
    write_model_bundle_manifest(
        manifest_path,
        mode="multihead_yolo",
        artifact_paths=artifacts,
        class_names=["color", "shape"],
    )

    bundle = discover_multihead_model_bundle(manifest_path)

    assert bundle is not None
    assert bundle["mode"] == "multihead_yolo"
    assert bundle["class_names"] == ["color", "shape"]
    assert sorted(bundle["artifact_paths"]) == sorted(str(p) for p in artifacts)


def test_discover_bundle_from_manifest_skips_when_artifacts_missing(
    tmp_path: Path,
) -> None:
    """A manifest pointing at missing artifacts must not silently load."""
    bundle_dir = tmp_path / "multihead_yolo_run"
    bundle_dir.mkdir()
    # Only create one of the two artifacts the manifest will reference.
    present = bundle_dir / "factor_0_yolo.pt"
    present.write_bytes(b"weights")
    missing_ref = bundle_dir / "factor_1_yolo.pt"

    manifest_path = bundle_dir / f"factor_0_yolo{MODEL_BUNDLE_MANIFEST_SUFFIX}"
    write_model_bundle_manifest(
        manifest_path,
        mode="multihead_yolo",
        artifact_paths=[present, missing_ref],
        class_names=[],
    )

    assert discover_multihead_model_bundle(manifest_path) is None


def test_discover_bundle_from_manifest_rejects_non_multihead(tmp_path: Path) -> None:
    """A flat (non-multi-head) bundle manifest should not be discovered."""
    bundle_dir = tmp_path / "flat_run"
    bundle_dir.mkdir()
    artifacts = _write_dummy_artifacts(bundle_dir, ["model_a.pt", "model_b.pt"])

    manifest_path = bundle_dir / f"flat{MODEL_BUNDLE_MANIFEST_SUFFIX}"
    raw = {
        "bundle_type": "classkit_model_bundle",
        "bundle_version": 1,
        "mode": "flat_yolo",
        "class_names": [],
        "artifacts": [{"path": p.name} for p in artifacts],
    }
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    assert discover_multihead_model_bundle(manifest_path) is None


def test_discover_bundle_falls_back_to_sibling_manifest(tmp_path: Path) -> None:
    """Selecting a .pt artifact still resolves through a sibling manifest."""
    bundle_dir = tmp_path / "multihead_yolo_run"
    bundle_dir.mkdir()
    artifacts = _write_dummy_artifacts(
        bundle_dir, ["factor_0_yolo.pt", "factor_1_yolo.pt"]
    )

    manifest_path = bundle_dir / f"factor_0_yolo{MODEL_BUNDLE_MANIFEST_SUFFIX}"
    write_model_bundle_manifest(
        manifest_path,
        mode="multihead_yolo",
        artifact_paths=artifacts,
        class_names=["letter"],
    )

    bundle = discover_multihead_model_bundle(artifacts[0])

    assert bundle is not None
    assert bundle["mode"] == "multihead_yolo"
    assert bundle["class_names"] == ["letter"]
    assert sorted(bundle["artifact_paths"]) == sorted(str(p) for p in artifacts)


def test_shared_trunk_role_publishes_single_artifact():
    """Service layer must not include the shared-trunk role in its multi-head
    manifest set: the .pth alone carries factor_names."""
    from hydra_suite.training import service as svc
    from hydra_suite.training.contracts import TrainingRole

    multihead_role_sets = [
        v
        for k, v in vars(svc).items()
        if isinstance(v, (set, frozenset))
        and any(
            getattr(item, "name", "").startswith("CLASSIFY_MULTIHEAD") for item in v
        )
    ]
    assert multihead_role_sets, "expected a multi-head role set in service.py"
    for role_set in multihead_role_sets:
        assert (
            TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM_SHARED not in role_set
        ), f"shared-trunk role must publish via single-artifact path; found in {role_set}"


def test_color_tag_preset_lists_shared_trunk_mode():
    from hydra_suite.classkit.config.presets import color_tag_preset

    scheme = color_tag_preset(2, ["red", "blue", "green"])
    assert "multihead_custom_shared" in scheme.training_modes
    scheme1 = color_tag_preset(1, ["red", "blue"])
    # Single-factor schemes still only allow flat modes
    assert "multihead_custom_shared" not in scheme1.training_modes


def test_task_workers_maps_shared_trunk_mode_to_role():
    from hydra_suite.classkit.jobs import task_workers
    from hydra_suite.training.contracts import TrainingRole

    mapping = getattr(task_workers, "MODE_TO_ROLE", None)
    if mapping is None:
        get_role = getattr(task_workers, "training_role_for_mode", None)
        if get_role is None:
            # Mapping currently lives on the GUI MainWindow as a static method.
            # Verify the role exists in that mapping instead.
            from hydra_suite.classkit.gui.main_window import MainWindow as _MW

            assert (
                _MW._training_role_for_mode("multihead_custom_shared")
                == TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM_SHARED
            )
            return
        assert (
            get_role("multihead_custom_shared")
            == TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM_SHARED
        )
    else:
        assert (
            mapping["multihead_custom_shared"]
            == TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM_SHARED
        )
