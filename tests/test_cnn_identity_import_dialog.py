"""Non-Qt logic in CNNIdentityImportDialog: metadata summary + scoring mode persistence."""

from __future__ import annotations


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


def test_describe_cnn_identity_candidate_multihead(tiny_multi_identity):
    from hydra_suite.trackerkit.gui.dialogs.cnn_identity_import_dialog import (
        describe_cnn_identity_candidate,
    )

    summary = describe_cnn_identity_candidate(str(tiny_multi_identity))
    assert summary["is_multihead"] is True
    assert summary["factor_names"] == ["color", "shape"]
    assert summary["class_names_per_factor"] == [["r", "g", "b"], ["sq", "ci"]]


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
