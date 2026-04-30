"""Non-Qt logic in CNNIdentityImportDialog: metadata summary + scoring mode persistence."""

from __future__ import annotations

import pytest


def test_describe_cnn_identity_candidate_flat(tiny_flat_subset):
    from hydra_suite.trackerkit.gui.dialogs.cnn_identity_import_dialog import (
        describe_cnn_identity_candidate,
    )

    summary = describe_cnn_identity_candidate(str(tiny_flat_subset))
    assert summary["arch"] == "tinyclassifier"
    assert summary["is_multihead"] is False
    assert summary["factor_names"] == ["flat"]
    assert summary["class_names_per_factor"] == [["left", "right"]]
    assert summary["input_size"] == (64, 64)
    assert summary["recommended_confidence_threshold"] is None


def test_describe_cnn_identity_candidate_multihead(tiny_multi_identity):
    from hydra_suite.trackerkit.gui.dialogs.cnn_identity_import_dialog import (
        describe_cnn_identity_candidate,
    )

    summary = describe_cnn_identity_candidate(str(tiny_multi_identity))
    assert summary["is_multihead"] is True
    assert summary["factor_names"] == ["color", "shape"]
    assert summary["class_names_per_factor"] == [["r", "g", "b"], ["sq", "ci"]]
    assert summary["recommended_confidence_threshold"] is None


def test_annotate_discovered_entry_writes_registry(
    tmp_path, monkeypatch, tiny_flat_subset
):
    """annotate_discovered_cnn_entry writes a registry entry referencing the managed path."""
    import shutil

    from hydra_suite.trackerkit.gui.dialogs.cnn_identity_import_dialog import (
        annotate_discovered_cnn_entry,
    )
    from hydra_suite.training import model_publish

    # Place the fixture under a managed publish root.
    models_root = tmp_path / "models"
    managed_rel = "tiny-classify/scheme/artifact.pth"
    (models_root / "tiny-classify/scheme").mkdir(parents=True)
    shutil.copy(str(tiny_flat_subset), str(models_root / managed_rel))
    monkeypatch.setattr(model_publish, "get_models_root", lambda: models_root)
    monkeypatch.setattr(
        model_publish,
        "_registry_path",
        lambda: models_root / "model_registry.json",
    )

    annotate_discovered_cnn_entry(
        rel_path=managed_rel,
        species="ant",
        classification_label="apriltag",
        scoring_mode="atomic",
    )

    entries = dict(model_publish.iter_registry_entries())
    assert managed_rel in entries
    m = entries[managed_rel]
    assert m["usage_role"] == "cnn_identity"
    assert m["scoring_mode"] == "atomic"
    assert m["species"] == "ant"
    assert m["classification_label"] == "apriltag"


def test_import_cnn_identity_candidate_copies_external_multihead_bundle(
    tmp_path, monkeypatch, tiny_flat_subset, tiny_flat_headtail
):
    import shutil

    from hydra_suite.trackerkit.gui.dialogs.cnn_identity_import_dialog import (
        import_cnn_identity_candidate,
    )
    from hydra_suite.training import model_publish

    models_root = tmp_path / "models"
    external_root = tmp_path / "external"
    external_root.mkdir()
    factor_a = external_root / "color.pth"
    factor_b = external_root / "heading.pth"
    shutil.copy2(str(tiny_flat_subset), str(factor_a))
    shutil.copy2(str(tiny_flat_headtail), str(factor_b))
    manifest = model_publish.write_classifier_multihead_manifest(
        external_root / "bundle.multihead.json",
        factor_entries=[
            {"factor": "side", "path": factor_a, "class_names": ["left", "right"]},
            {
                "factor": "heading",
                "path": factor_b,
                "class_names": ["up", "down", "left", "right", "unknown"],
            },
        ],
        input_size=(64, 64),
        monochrome=False,
        recommended_confidence_threshold=0.72,
    )

    monkeypatch.setattr(model_publish, "get_models_root", lambda: models_root)
    monkeypatch.setattr(
        model_publish,
        "_registry_path",
        lambda: models_root / "model_registry.json",
    )

    rel_path = import_cnn_identity_candidate(
        model_path=str(manifest),
        species="ant",
        classification_label="colortag",
        scoring_mode="per_head_average",
    )

    stored_manifest = models_root / rel_path
    assert stored_manifest.exists()
    factor_prefix = stored_manifest.name.removesuffix(".multihead.json")
    copied_factor_paths = sorted(stored_manifest.parent.glob(f"{factor_prefix}_*.pth"))
    assert len(copied_factor_paths) == 2
    entries = dict(model_publish.iter_registry_entries())
    meta = entries[rel_path]
    assert meta["usage_role"] == "cnn_identity"
    assert meta["scoring_mode"] == "per_head_average"
    assert meta["factor_names"] == ["side", "heading"]
    assert meta["recommended_confidence_threshold"] == pytest.approx(0.72)
